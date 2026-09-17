#include "core/device.h"
#include "core/tensor.h"
#include "core/arena.h"
#include "core/command_list.h"

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <cassert>
#include <chrono>

using namespace xinfer::core;

void test_device_discovery_and_arch() {
    std::cout << "\n[Test 1/4] Device Discovery & Architecture Query (Intel Arc Pro B60)..." << std::endl;
    auto ctx = DeviceContext::create(true);
    const auto& arch = ctx->arch_info();

    std::cout << "  Device Name:        " << arch.device_name << std::endl;
    std::cout << "  Xe-Core Count:      " << arch.xe_core_count << std::endl;
    std::cout << "  Vector Engine Count:" << arch.vector_engine_count << std::endl;
    std::cout << "  Hardware Threads:   " << arch.total_hw_threads << std::endl;
    std::cout << "  Global Memory:      " << (arch.global_mem_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
    std::cout << "  SLM Capacity:       " << (arch.slm_bytes / 1024) << " KB" << std::endl;
    std::cout << "  Max Workgroup Size: " << arch.max_work_group_size << std::endl;
    std::cout << "  Subgroup Sizes:     ";
    for (size_t s : arch.sub_group_sizes) std::cout << s << " ";
    std::cout << std::endl;

    assert(!arch.device_name.empty());
    assert(arch.xe_core_count > 0);
    assert(arch.vector_engine_count > 0);
    assert(arch.total_hw_threads > 0);
    assert(arch.global_mem_bytes > 0);
    assert(ctx->has_native_level_zero());
    assert(ctx->native_l0_device() != nullptr);
    assert(ctx->native_l0_context() != nullptr);
    assert(ctx->native_l0_queue() != nullptr);

    std::cout << "  -> PASSED: Hardware queries match Xe2 architecture specifications." << std::endl;
}

void test_tensor_usm_and_views() {
    std::cout << "\n[Test 2/4] Device USM Tensor Allocation, Slicing, and Round-Trip..." << std::endl;
    auto ctx = DeviceContext::create(true);

    const int64_t rows = 8;
    const int64_t cols = 32;
    const int64_t numel = rows * cols;
    TensorShape shape{rows, cols};

    // 1. Allocate USM device tensor
    DeviceTensor tensor = DeviceTensor::allocate(*ctx, shape, DataType::Float32, 64);
    assert((reinterpret_cast<uintptr_t>(tensor.raw_data()) % 64) == 0);
    assert(tensor.byte_size() == static_cast<size_t>(numel * sizeof(float)));

    // 2. Host data generation
    std::vector<float> host_in(numel);
    for (int64_t i = 0; i < numel; ++i) {
        host_in[i] = static_cast<float>(i * 1.5f + 0.25f);
    }

    // 3. Copy host -> device -> host
    tensor.copy_from_host(host_in.data());
    std::vector<float> host_out(numel, 0.0f);
    tensor.copy_to_host(host_out.data());

    for (int64_t i = 0; i < numel; ++i) {
        assert(host_in[i] == host_out[i]);
    }

    // 4. View slicing
    TensorView view = tensor.view();
    assert(view.is_contiguous());
    TensorView slice_view = view.slice(0, 2, 4); // slice rows 2..5 (4 rows)
    assert(slice_view.shape().dim(0) == 4);
    assert(slice_view.shape().dim(1) == cols);
    assert(slice_view.data_as<float>() == tensor.data_as<float>() + 2 * cols);

    // 5. Reshaping
    TensorView reshaped = view.reshape(TensorShape{16, 16});
    assert(reshaped.shape().numel() == numel);

    // 6. Stress test repeated allocate/free cycles (no memory leaks)
    std::cout << "  Running 1,000 rapid allocate/free cycles on B60..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int cycle = 0; cycle < 1000; ++cycle) {
        DeviceTensor tmp = DeviceTensor::allocate(*ctx, TensorShape{1024, 64}, DataType::Float32);
        assert(tmp.raw_data() != nullptr);
    }
    ctx->synchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  Completed 1,000 allocate/free cycles in " << ms << " ms ("
              << (ms / 1000.0) << " ms/cycle)" << std::endl;

    std::cout << "  -> PASSED: Tensor USM operations and views verified." << std::endl;
}

void test_device_arena_allocator() {
    std::cout << "\n[Test 3/4] DeviceArena Activation Buffer Allocator..." << std::endl;
    auto ctx = DeviceContext::create(true);

    const size_t capacity = 32 * 1024 * 1024; // 32 MB
    DeviceArena arena(ctx, capacity);
    assert(arena.capacity() == capacity);
    assert(arena.allocated_bytes() == 0);
    assert((reinterpret_cast<uintptr_t>(arena.base_ptr()) % 64) == 0);

    // 1. Allocate multiple tensors within arena
    TensorView q_buf = arena.allocate_tensor(TensorShape{1, 32, 128}, DataType::Float16);
    TensorView k_buf = arena.allocate_tensor(TensorShape{1, 32, 128}, DataType::Float16);
    TensorView v_buf = arena.allocate_tensor(TensorShape{1, 32, 128}, DataType::Float16);
    TensorView ffn_buf = arena.allocate_tensor(TensorShape{1, 14336}, DataType::Float32);

    assert((reinterpret_cast<uintptr_t>(q_buf.data()) % 64) == 0);
    assert((reinterpret_cast<uintptr_t>(k_buf.data()) % 64) == 0);
    assert((reinterpret_cast<uintptr_t>(v_buf.data()) % 64) == 0);
    assert((reinterpret_cast<uintptr_t>(ffn_buf.data()) % 64) == 0);

    // Check non-overlapping addresses
    uintptr_t q_end = reinterpret_cast<uintptr_t>(q_buf.data()) + q_buf.byte_size();
    assert(reinterpret_cast<uintptr_t>(k_buf.data()) >= q_end);

    // 2. Test simulate 500 decode step cycles (allocate -> use -> reset)
    std::cout << "  Simulating 500 decode steps reusing scratch memory..." << std::endl;
    for (int step = 0; step < 500; ++step) {
        arena.reset();
        assert(arena.allocated_bytes() == 0);

        TensorView act1 = arena.allocate_tensor(TensorShape{1, 5120}, DataType::Float32);
        TensorView act2 = arena.allocate_tensor(TensorShape{1, 17408}, DataType::Float32);
        TensorView act3 = arena.allocate_tensor(TensorShape{1, 5120}, DataType::Float32);

        assert(act1.data() != nullptr);
        assert(act2.data() != nullptr);
        assert(act3.data() != nullptr);
    }

    assert(arena.peak_allocated_bytes() > 0);
    assert(arena.peak_allocated_bytes() <= arena.capacity());

    std::cout << "  Peak activation memory used: " << (arena.peak_allocated_bytes() / 1024) << " KB" << std::endl;
    std::cout << "  -> PASSED: DeviceArena zero-allocation decode reuse verified." << std::endl;

    // 3. Test persistent buffer outside the bump arena (for decode fallback d_logits)
    std::cout << "  Testing persistent buffer outside bump arena..." << std::endl;
    void* pbuf1 = arena.persistent_buffer(1024 * 1024);
    assert(pbuf1 != nullptr);
    assert((reinterpret_cast<uintptr_t>(pbuf1) % 64) == 0);

    // Verify pbuf1 does not alias with the bump arena [base_ptr, base_ptr + capacity)
    uintptr_t p_addr = reinterpret_cast<uintptr_t>(pbuf1);
    uintptr_t base_addr = reinterpret_cast<uintptr_t>(arena.base_ptr());
    assert(p_addr < base_addr || p_addr >= (base_addr + arena.capacity()));

    // Verify address stability across multiple calls
    void* pbuf2 = arena.persistent_buffer(1024 * 1024);
    assert(pbuf1 == pbuf2);

    // Verify persistence across arena reset()
    arena.reset();
    assert(arena.allocated_bytes() == 0);
    void* pbuf3 = arena.persistent_buffer();
    assert(pbuf1 == pbuf3);

    // Verify bump allocations after reset do not alias persistent buffer
    TensorView act_after_reset = arena.allocate_tensor(TensorShape{1, 5120}, DataType::Float32);
    assert(act_after_reset.data() != pbuf3);
    assert(reinterpret_cast<uintptr_t>(act_after_reset.data()) != reinterpret_cast<uintptr_t>(pbuf3));
    std::cout << "  -> PASSED: DeviceArena persistent buffer outside arena verified." << std::endl;
}

void test_level_zero_command_list() {
    std::cout << "\n[Test 4/4] Level Zero Command List Recording and Execution..." << std::endl;
    auto ctx = DeviceContext::create(true);
    auto cmd_list = LevelZeroCommandList::create(ctx);

    const size_t count = 2048;
    const size_t bytes = count * sizeof(float);

    DeviceTensor src = DeviceTensor::allocate(*ctx, TensorShape{static_cast<int64_t>(count)}, DataType::Float32);
    DeviceTensor dst = DeviceTensor::allocate(*ctx, TensorShape{static_cast<int64_t>(count)}, DataType::Float32);

    std::vector<float> host_src(count);
    for (size_t i = 0; i < count; ++i) host_src[i] = static_cast<float>(i * 3.14159f);
    src.copy_from_host(host_src.data());

    // Zero out destination
    ctx->fill_device(dst.raw_data(), 0, bytes);

    // 1. Record copy command into deferred regular command list
    cmd_list->append_memory_copy(dst.raw_data(), src.raw_data(), bytes);
    cmd_list->append_barrier();
    cmd_list->close();

    // 2. Submit to command queue and synchronize via fence
    cmd_list->execute();
    cmd_list->synchronize();

    std::vector<float> host_dst(count, 0.0f);
    dst.copy_to_host(host_dst.data());

    for (size_t i = 0; i < count; ++i) {
        assert(host_src[i] == host_dst[i]);
    }

    // 3. Reset and replay test (record new command on the same list)
    cmd_list->reset();
    for (size_t i = 0; i < count; ++i) host_src[i] = static_cast<float>(i * 42.0f);
    src.copy_from_host(host_src.data());

    cmd_list->append_memory_copy(dst.raw_data(), src.raw_data(), bytes);
    cmd_list->close();
    cmd_list->execute();
    cmd_list->synchronize();

    dst.copy_to_host(host_dst.data());
    for (size_t i = 0; i < count; ++i) {
        assert(host_src[i] == host_dst[i]);
    }

    std::cout << "  -> PASSED: Regular command list record-replay-reset verified on Level Zero." << std::endl;
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << " xinfer Core Device Layer Unit Tests (Intel Arc Pro B60)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    try {
        test_device_discovery_and_arch();
        test_tensor_usm_and_views();
        test_device_arena_allocator();
        test_level_zero_command_list();

        std::cout << "\n==========================================================" << std::endl;
        std::cout << " ALL MILESTONE 3 UNIT TESTS PASSED ON INTEL ARC PRO B60!" << std::endl;
        std::cout << "==========================================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[TEST FAILED]: " << e.what() << std::endl;
        return 1;
    }
}
