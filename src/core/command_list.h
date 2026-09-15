#pragma once

#include "device.h"
#include <memory>
#include <cstdint>
#include <string>

namespace xinfer::core {

class LevelZeroCommandList {
public:
    static std::unique_ptr<LevelZeroCommandList> create(std::shared_ptr<DeviceContext> ctx);

    ~LevelZeroCommandList();

    LevelZeroCommandList(const LevelZeroCommandList&) = delete;
    LevelZeroCommandList& operator=(const LevelZeroCommandList&) = delete;

    LevelZeroCommandList(LevelZeroCommandList&& other) noexcept;
    LevelZeroCommandList& operator=(LevelZeroCommandList&& other) noexcept;

    // Record memory copy into command list
    void append_memory_copy(void* dst_device, const void* src_device, size_t bytes);

    // Record execution barrier
    void append_barrier();

    // Close command list, making it ready for execution
    void close();

    // Reset command list back to open state for recording new commands
    void reset();

    // Submit closed command list to queue
    void execute();

    // Block host until submitted execution finishes
    void synchronize();

    bool is_closed() const noexcept { return is_closed_; }
    void* handle() const noexcept { return cmd_list_; }

private:
    explicit LevelZeroCommandList(std::shared_ptr<DeviceContext> ctx);
    void init();

    std::shared_ptr<DeviceContext> ctx_;
    void* h_module_{nullptr};
    void* cmd_list_{nullptr};
    void* cmd_queue_{nullptr};
    void* fence_{nullptr};
    bool  is_closed_{false};

    // Internal function pointers loaded from ze_loader.dll
    struct L0DispatchTable;
    std::unique_ptr<L0DispatchTable> fn_;
};

} // namespace xinfer::core
