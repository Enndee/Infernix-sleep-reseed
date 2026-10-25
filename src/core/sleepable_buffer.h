#pragma once

#include "core/arena.h"
#include "core/vmm_arena.h"

#include <cuda_runtime.h>

#include <memory>

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

} // namespace infernix