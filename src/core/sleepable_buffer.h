#pragma once

#include "core/arena.h"
#include "core/vmm_arena.h"

#include <cuda_runtime.h>

#include <memory>
#include <functional>
#include <utility>
#include <vector>

namespace infernix {

// Physical backing may sleep; the virtual address and every non-owning view stay stable.
// The caller serializes transitions and quiesces ALL streams before sleep(). Immutable
// contents are captured once after initialization; scratch has no mirror and is disposable.
class SleepableBuffer {
public:
    SleepableBuffer(std::size_t bytes, int device);
    [[nodiscard]] DeviceSpan span() const noexcept { return {arena_.base(), bytes_}; }
    [[nodiscard]] std::size_t mapped_bytes() const noexcept { return arena_.mapped_bytes(); }
    [[nodiscard]] bool asleep() const noexcept { return asleep_; }
    void capture(cudaStream_t stream);
    void sleep();
    // On allocation failure, roll back the newly mapped chunks and remain asleep.
    // No GPU work is allowed until all essential buffers have successfully woken.
    [[nodiscard]] bool wake(cudaStream_t stream);

private:
    std::size_t bytes_;
    VmmArena arena_;
    std::unique_ptr<PinnedHostBuffer> mirror_;
    bool asleep_ = false;
};

// Mutable, sparsely mapped storage (KV planes and execution tables). The owner must
// quiesce every reader/writer before sleep and forbid access until wake succeeds.
// Unlike immutable weights, the mirror is refreshed on EVERY sleep. The range's
// mappings and all byte contents are restored, without changing page ownership or VA.
class VmmRangeSnapshot {
public:
    // Optional allocator permits deterministic fault injection without exhausting
    // the user's GPU. It must have VmmRange::map's all-or-nothing chunk contract.
    using MapChunk = std::function<bool(std::size_t)>;
    explicit VmmRangeSnapshot(VmmRange& range, MapChunk map_chunk = {})
        : range_(range), map_chunk_(std::move(map_chunk)) {}
    void sleep(cudaStream_t stream);
    [[nodiscard]] bool wake(cudaStream_t stream);
    [[nodiscard]] bool asleep() const noexcept { return asleep_; }
    [[nodiscard]] std::size_t saved_bytes() const noexcept { return chunks_.size() * range_.chunk_bytes(); }

private:
    void release_mappings();
    void copy_snapshot(cudaStream_t stream, bool to_host);
    VmmRange& range_;
    MapChunk map_chunk_;
    std::vector<std::size_t> chunks_;
    // Bound each pinned allocation: large single cudaHostAlloc calls can stall
    // under WDDM. The aggregate high-water mirror is reused across sleeps.
    static constexpr std::size_t kMirrorChunkBytes = 256ULL << 20;
    std::vector<std::unique_ptr<PinnedHostBuffer>> mirrors_;
    bool asleep_ = false;
};

} // namespace infernix