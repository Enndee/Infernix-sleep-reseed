#include "core/sleepable_buffer.h"

#include "core/cuda_vmm.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace infernix {
namespace {
void check(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("sleepable buffer: ") + cudaGetErrorString(error));
    }
}
} // namespace

SleepableBuffer::SleepableBuffer(std::size_t bytes, int device)
    : bytes_(bytes), arena_(device, bytes, std::max<std::size_t>(16ULL << 20, vmm::granularity(device))) {
    while (arena_.mapped_bytes() < bytes_) {
        if (!arena_.map_chunk()) { throw std::runtime_error("sleepable buffer: no device memory at startup"); }
    }
}

void SleepableBuffer::capture(cudaStream_t stream) {
    if (asleep_) { throw std::logic_error("cannot capture a sleeping buffer"); }
    if (mirror_) { return; }
    auto mirror = std::make_unique<PinnedHostBuffer>(bytes_);
    check(cudaMemcpyAsync(mirror->data(), arena_.base(), bytes_, cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    mirror_ = std::move(mirror);
}

void SleepableBuffer::sleep() {
    if (asleep_) { return; }
    // Even a failed partial unmap must go through wake before any GPU access.
    asleep_ = true;
    while (arena_.mapped_bytes() != 0) { arena_.unmap_chunk(); }
}

bool SleepableBuffer::wake(cudaStream_t stream) {
    if (!asleep_) { return true; }
    try {
        while (arena_.mapped_bytes() < bytes_) {
            if (!arena_.map_chunk()) {
                while (arena_.mapped_bytes() != 0) { arena_.unmap_chunk(); }
                return false;
            }
        }
        if (mirror_) {
            check(cudaMemcpyAsync(arena_.base(), mirror_->data(), bytes_, cudaMemcpyHostToDevice, stream));
            check(cudaStreamSynchronize(stream));
        }
    } catch (...) {
        // Settle any queued transfer before removing its destination.
        (void)cudaStreamSynchronize(stream);
        while (arena_.mapped_bytes() != 0) { arena_.unmap_chunk(); }
        throw;
    }
    asleep_ = false;
    return true;
}

} // namespace infernix