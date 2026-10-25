#include "core/sleepable_buffer.h"
#include "core/decode_graph.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || count == 0) { return 77; }
    CUDA_CHECK(status);
    CUDA_CHECK(cudaSetDevice(0));
    if (!infernix::VmmArena::supported(0)) { return 77; }
    try {
        constexpr std::size_t bytes = (16ULL << 20) + 17; // rounding and multiple physical chunks
        infernix::SleepableBuffer weights(bytes, 0), scratch(4096, 0);
        const auto original = weights.span();
        const auto original_scratch = scratch.span();
        CUDA_CHECK(cudaMemset(original.data, 0x5a, bytes));
        weights.capture(nullptr);
        CUDA_CHECK(cudaMemset(original.data, 0xa7, bytes));
        weights.capture(nullptr); // the immutable mirror is captured only once
        CUDA_CHECK(cudaDeviceSynchronize()); // capture-once is a no-op; settle the default-stream overwrite
        infernix::DeviceBuffer output(4096);
        cudaStream_t stream{};
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        infernix::DecodeGraphDefinition definition;
        definition.capture(stream, [&] {
            CUDA_CHECK(cudaMemsetAsync(original_scratch.data, 0x31, 4096, stream));
            CUDA_CHECK(cudaMemcpyAsync(output.p, original.data, 4096, cudaMemcpyDeviceToDevice, stream));
        });
        infernix::DecodeGraphExecutable graph;
        graph.instantiate(definition);
        for (int cycle = 0; cycle != 5; ++cycle) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            weights.sleep();
            scratch.sleep();
            weights.sleep(); // idempotent, including destruction while asleep below
            if (weights.mapped_bytes() != 0 || scratch.mapped_bytes() != 0 || !weights.asleep()) {
                throw std::runtime_error("physical backing not released");
            }
            bool refused = false;
            try { scratch.capture(stream); } catch (const std::logic_error&) { refused = true; }
            if (!refused) { throw std::runtime_error("capture accepted unmapped storage"); }
            if (!weights.wake(stream) || !scratch.wake(stream) || !weights.wake(stream)) {
                throw std::runtime_error("wake allocation failed");
            }
            if (weights.span().data != original.data || scratch.span().data != original_scratch.data ||
                weights.mapped_bytes() < bytes) { throw std::runtime_error("address/capacity changed"); }
            graph.launch(stream); // executable created BEFORE unmap/remap: no recapture
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::array<unsigned char, 4096> got{};
            output.copy_to_host(got.data(), got.size());
            for (auto value : got) {
                if (value != 0x5a) { throw std::runtime_error("graph read changed restored weights"); }
            }
            CUDA_CHECK(cudaMemcpy(got.data(), original_scratch.data, got.size(), cudaMemcpyDeviceToHost));
            for (auto value : got) {
                if (value != 0x31) { throw std::runtime_error("graph failed to write restored scratch"); }
            }
            std::array<unsigned char, 17> tail{};
            CUDA_CHECK(cudaMemcpy(tail.data(), static_cast<unsigned char*>(original.data) + bytes - tail.size(),
                                  tail.size(), cudaMemcpyDeviceToHost));
            for (auto value : tail) {
                if (value != 0x5a) { throw std::runtime_error("mirror lost final partial chunk"); }
            }
            std::vector<unsigned char> restored(bytes);
            CUDA_CHECK(cudaMemcpy(restored.data(), original.data, bytes, cudaMemcpyDeviceToHost));
            for (auto value : restored) {
                if (value != 0x5a) { throw std::runtime_error("mirror lost bytes inside restored chunks"); }
            }
        }
        graph.reset();
        definition.reset();
        CUDA_CHECK(cudaStreamDestroy(stream));
        weights.sleep();
        scratch.sleep();
        std::cout << "5 sleep/wake cycles: immutable bytes, scratch writes, stable pointers and captured graph PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}