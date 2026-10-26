#include "core/sleepable_buffer.h"
#include "core/decode_graph.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <array>
#include <algorithm>
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
        // Sparse KV-like planes and distant metadata, refreshed EVERY sleep.
        constexpr std::size_t chunk = 2ULL << 20;
        infernix::VmmRange range(0, 9 * chunk, chunk);
        int fail_after = -1;
        bool throw_failure = false;
        infernix::VmmRangeSnapshot snapshot(range, [&](std::size_t c) {
            if (fail_after == 0) {
                if (throw_failure) { throw std::runtime_error("injected map failure"); }
                return false;
            }
            if (fail_after > 0) { --fail_after; }
            return range.map(c);
        });
        const auto base = range.base();
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        infernix::DecodeGraphDefinition kv_definition;
        infernix::DecodeGraphExecutable kv_graph;
        snapshot.sleep(nullptr);
        snapshot.sleep(nullptr);
        if (snapshot.saved_bytes() || !snapshot.wake(nullptr)) { throw std::runtime_error("empty snapshot failed"); }
        for (int cycle = 0; cycle != 5; ++cycle) {
            std::vector<std::size_t> active{0, 3, 8};
            if (cycle >= 2) { active.push_back(5); }
            if (cycle >= 4) { active.erase(active.begin() + 1); }
            for (std::size_t c = 0; c < range.chunk_count(); ++c) {
                const bool needed = std::find(active.begin(), active.end(), c) != active.end();
                if (needed && !range.mapped(c) && !range.map(c)) { throw std::runtime_error("test mapping failed"); }
                if (!needed && range.mapped(c)) { range.unmap(c); }
            }
            std::vector<unsigned char> expected(chunk);
            for (std::size_t i = 0; i < expected.size(); ++i) {
                expected[i] = static_cast<unsigned char>((i * 17 + cycle * 43) % 251);
            }
            for (const auto c : active) {
                CUDA_CHECK(cudaMemcpy(static_cast<std::byte*>(base) + c * chunk, expected.data(), chunk,
                                      cudaMemcpyHostToDevice));
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            if (cycle == 0) {
                kv_definition.capture(stream, [&] {
                    CUDA_CHECK(cudaMemcpyAsync(output.p, base, 4096, cudaMemcpyDeviceToDevice, stream));
                });
                kv_graph.instantiate(kv_definition);
            }
            snapshot.sleep(nullptr);
            snapshot.sleep(nullptr);
            if (range.mapped_bytes() != 0 || snapshot.saved_bytes() != active.size() * chunk) {
                throw std::runtime_error("snapshot did not release all physical KV backing");
            }
            if (cycle == 1 || cycle == 3) {
                fail_after = 2; // fail AFTER some chunks have been restored
                throw_failure = cycle == 3;
                bool failed = false;
                try { failed = !snapshot.wake(nullptr); }
                catch (const std::runtime_error&) {
                    if (!throw_failure) { throw; }
                    failed = true;
                }
                if (!failed || !snapshot.asleep() || range.mapped_bytes() != 0) {
                    throw std::runtime_error("partial restore did not roll back");
                }
                fail_after = -1;
            }
            if (!snapshot.wake(nullptr) || !snapshot.wake(nullptr) || range.base() != base) {
                throw std::runtime_error("snapshot wake changed address");
            }
            for (std::size_t c = 0; c < range.chunk_count(); ++c) {
                const bool needed = std::find(active.begin(), active.end(), c) != active.end();
                if (range.mapped(c) != needed) { throw std::runtime_error("snapshot changed sparse mappings"); }
                if (needed) {
                    std::vector<unsigned char> got(chunk);
                    CUDA_CHECK(cudaMemcpy(got.data(), static_cast<std::byte*>(base) + c * chunk, chunk,
                                          cudaMemcpyDeviceToHost));
                    if (got != expected) { throw std::runtime_error("snapshot restored stale/corrupt mutable state"); }
                }
            }
            kv_graph.launch(stream); // the graph predates every unmap/remap
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::array<unsigned char, 4096> got_graph{};
            output.copy_to_host(got_graph.data(), got_graph.size());
            if (!std::equal(got_graph.begin(), got_graph.end(), expected.begin())) {
                throw std::runtime_error("captured graph read stale restored KV");
            }
        }
        kv_graph.reset();
        kv_definition.reset();
        CUDA_CHECK(cudaStreamDestroy(stream));
        snapshot.sleep(nullptr);
        // Cross the bounded host-mirror slab boundary, including its partial tail.
        infernix::VmmRange big(0, 130 * chunk, chunk);
        for (std::size_t c = 0; c < big.chunk_count(); ++c) {
            if (!big.map(c)) { throw std::runtime_error("large snapshot test mapping failed"); }
        }
        infernix::VmmRangeSnapshot big_snapshot(big);
        for (int cycle = 0; cycle < 2; ++cycle) {
            const auto value = static_cast<unsigned char>(0x51 + cycle);
            CUDA_CHECK(cudaMemset(big.base(), value, big.mapped_bytes()));
            CUDA_CHECK(cudaDeviceSynchronize());
            big_snapshot.sleep(nullptr);
            if (big.mapped_bytes() || !big_snapshot.wake(nullptr)) { throw std::runtime_error("slab restore failed"); }
            std::vector<unsigned char> got(big.mapped_bytes());
            CUDA_CHECK(cudaMemcpy(got.data(), big.base(), got.size(), cudaMemcpyDeviceToHost));
            if (!std::all_of(got.begin(), got.end(), [value](auto b) { return b == value; })) {
                throw std::runtime_error("slab boundary/tail data corruption");
            }
        }
        std::cout << "5 sleep/wake cycles: immutable bytes, scratch writes, stable pointers and captured graph PASS\n";
        std::cout << "Mutable KV: fresh bytes, sparse growth/shrink, captured graph, allocation/exception rollback and retry, multi-slab mirror PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}