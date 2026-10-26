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

void VmmRangeSnapshot::sleep(cudaStream_t stream) {
    if (asleep_) { return; }
    std::vector<std::size_t> chunks;
    for (std::size_t c = 0; c < range_.chunk_count(); ++c) {
        if (range_.mapped(c)) { chunks.push_back(c); }
    }
    const auto bytes = chunks.size() * range_.chunk_bytes();
    // Allocate before touching any mapping. Retain the high-water mirror to avoid
    // repeated host registration cost; allocation failure leaves the device intact.
    const auto per_mirror = std::max<std::size_t>(1, kMirrorChunkBytes / range_.chunk_bytes());
    const auto mirror_bytes = per_mirror * range_.chunk_bytes();
    const auto needed_mirrors = (chunks.size() + per_mirror - 1) / per_mirror;
    for (std::size_t m = 0; m < needed_mirrors; ++m) {
        const auto needed = std::min(mirror_bytes, bytes - m * mirror_bytes);
        if (m >= mirrors_.size()) { mirrors_.push_back(std::make_unique<PinnedHostBuffer>(needed)); }
        else if (mirrors_[m]->size() < needed) { mirrors_[m] = std::make_unique<PinnedHostBuffer>(needed); }
    }
    chunks_ = std::move(chunks);
    try {
        copy_snapshot(stream, true);
        check(cudaStreamSynchronize(stream));
    } catch (...) {
        (void)cudaStreamSynchronize(stream);
        throw;
    }
    asleep_ = true; // a partial unmap must also pass through restore before reuse
    release_mappings();
}

void VmmRangeSnapshot::release_mappings() {
    for (const auto c : chunks_) {
        if (range_.mapped(c)) { range_.unmap(c); }
    }
}

void VmmRangeSnapshot::copy_snapshot(cudaStream_t stream, bool to_host) {
    const auto per_mirror = std::max<std::size_t>(1, kMirrorChunkBytes / range_.chunk_bytes());
    // Merge adjacent device chunks, but never cross a host slab or a sparse hole.
    // Large KV planes should not pay thousands of tiny DMA submissions.
    for (std::size_t i = 0; i < chunks_.size();) {
        std::size_t end = i + 1;
        while (end < chunks_.size() && end / per_mirror == i / per_mirror &&
               chunks_[end] == chunks_[end - 1] + 1) { ++end; }
        auto* host = static_cast<std::byte*>(mirrors_[i / per_mirror]->data()) +
                     (i % per_mirror) * range_.chunk_bytes();
        auto* device = static_cast<std::byte*>(range_.base()) + chunks_[i] * range_.chunk_bytes();
        check(cudaMemcpyAsync(to_host ? host : device, to_host ? device : host,
                              (end - i) * range_.chunk_bytes(),
                              to_host ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice, stream));
        i = end;
    }
}

bool VmmRangeSnapshot::wake(cudaStream_t stream) {
    if (!asleep_) { return true; }
    try {
        for (const auto c : chunks_) {
            if (!range_.mapped(c) && !(map_chunk_ ? map_chunk_(c) : range_.map(c))) {
                release_mappings();
                return false;
            }
        }
        copy_snapshot(stream, false);
        check(cudaStreamSynchronize(stream));
    } catch (...) {
        (void)cudaStreamSynchronize(stream);
        release_mappings();
        throw;
    }
    asleep_ = false;
    return true;
}

} // namespace infernix