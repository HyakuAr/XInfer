#include "device.h"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <cstdlib>

namespace xinfer::core {

std::shared_ptr<DeviceContext> DeviceContext::create(bool prefer_b60) {
    auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.empty()) {
        throw std::runtime_error("No SYCL GPU devices found on the system.");
    }

    sycl::device selected_device = gpus[0];
    bool found = false;

    if (prefer_b60) {
        for (const auto& dev : gpus) {
            std::string name = dev.get_info<sycl::info::device::name>();
            auto backend = dev.get_backend();
            if (name.find("B60") != std::string::npos && backend == sycl::backend::ext_oneapi_level_zero) {
                selected_device = dev;
                found = true;
                break;
            }
        }
    }

    if (!found) {
        for (const auto& dev : gpus) {
            if (dev.get_backend() == sycl::backend::ext_oneapi_level_zero) {
                selected_device = dev;
                found = true;
                break;
            }
        }
    }

    if (!found) {
        std::cerr << "[xinfer::DeviceContext] Warning: No "
                  << (prefer_b60 ? "Intel Arc Pro B60 or " : "")
                  << "Level-Zero GPU device found. "
                  << "Falling back to non-preferred, non-Level-Zero device '"
                  << selected_device.get_info<sycl::info::device::name>()
                  << "'. Level-Zero backend features will be unavailable." << std::endl;
    }

    return std::make_shared<DeviceContext>(selected_device);
}

DeviceContext::DeviceContext(const sycl::device& dev)
    : queue_(dev, sycl::property_list{
          sycl::property::queue::in_order{},
          sycl::property::queue::enable_profiling{}
      }) {
    query_arch_info();
}

DeviceContext::~DeviceContext() {
    synchronize();
}

void DeviceContext::query_arch_info() {
    auto dev = device();
    arch_info_.device_name = dev.get_info<sycl::info::device::name>();
    arch_info_.global_mem_bytes = dev.get_info<sycl::info::device::global_mem_size>();
    arch_info_.slm_bytes = static_cast<uint32_t>(dev.get_info<sycl::info::device::local_mem_size>());
    arch_info_.max_work_group_size = dev.get_info<sycl::info::device::max_work_group_size>();
    arch_info_.sub_group_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();

    // Query Xe2 hardware architecture metrics as specified in docs/vendor/xe-gpu-architecture.md
    try {
        uint32_t slices = dev.get_info<sycl::ext::intel::info::device::gpu_slices>();
        uint32_t subslices = dev.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
        uint32_t eus = dev.get_info<sycl::ext::intel::info::device::gpu_eu_count_per_subslice>();
        uint32_t threads = dev.get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>();

        arch_info_.xe_core_count = slices * subslices;
        arch_info_.vector_engine_count = arch_info_.xe_core_count * eus;
        arch_info_.hw_threads_per_ve = threads;
        arch_info_.total_hw_threads = arch_info_.vector_engine_count * threads;
    } catch (const std::exception& e) {
        std::cerr << "[xinfer::DeviceContext] Warning: Failed to query Xe hardware architecture metrics ("
                  << e.what() << "). Falling back to default B60 architecture parameters." << std::endl;
        // Fallback default for Xe2-HPG Battlemage B60 (20 Xe-Cores, 8 VE/core, 8 hw threads/VE)
        arch_info_.xe_core_count = 20;
        arch_info_.vector_engine_count = 160;
        arch_info_.hw_threads_per_ve = 8;
        arch_info_.total_hw_threads = 1280;
    } catch (...) {
        std::cerr << "[xinfer::DeviceContext] Warning: Failed to query Xe hardware architecture metrics (unknown error). "
                  << "Falling back to default B60 architecture parameters." << std::endl;
        // Fallback default for Xe2-HPG Battlemage B60 (20 Xe-Cores, 8 VE/core, 8 hw threads/VE)
        arch_info_.xe_core_count = 20;
        arch_info_.vector_engine_count = 160;
        arch_info_.hw_threads_per_ve = 8;
        arch_info_.total_hw_threads = 1280;
    }
}

bool DeviceContext::has_native_level_zero() const noexcept {
    return device().get_backend() == sycl::backend::ext_oneapi_level_zero;
}

void* DeviceContext::native_l0_device() const {
    if (!has_native_level_zero()) return nullptr;
    return reinterpret_cast<void*>(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device()));
}

void* DeviceContext::native_l0_context() const {
    if (!has_native_level_zero()) return nullptr;
    return reinterpret_cast<void*>(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(context()));
}

void* DeviceContext::native_l0_queue() const {
    if (!has_native_level_zero()) return nullptr;
    auto native_q = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue_);
    if (std::holds_alternative<ze_command_queue_handle_t>(native_q)) {
        return reinterpret_cast<void*>(std::get<ze_command_queue_handle_t>(native_q));
    } else if (std::holds_alternative<ze_command_list_handle_t>(native_q)) {
        return reinterpret_cast<void*>(std::get<ze_command_list_handle_t>(native_q));
    }
    return nullptr;
}

void* DeviceContext::allocate_device(size_t bytes, size_t alignment) {
    if (bytes == 0) return nullptr;
    void* ptr = sycl::aligned_alloc_device(alignment, bytes, queue_);
    if (!ptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

void DeviceContext::free_device(void* ptr) {
    if (ptr) {
        sycl::free(ptr, queue_);
    }
}

void* DeviceContext::allocate_host(size_t bytes, size_t alignment) {
    if (bytes == 0) return nullptr;
    void* ptr = sycl::aligned_alloc_host(alignment, bytes, queue_);
    if (!ptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

void DeviceContext::free_host(void* ptr) {
    if (ptr) {
        sycl::free(ptr, queue_);
    }
}

void DeviceContext::copy_host_to_device(void* dst_device, const void* src_host, size_t bytes, bool blocking) {
    if (bytes == 0 || !dst_device || !src_host) return;
    auto event = queue_.memcpy(dst_device, src_host, bytes);
    if (blocking) {
        event.wait();
    }
}

void DeviceContext::copy_device_to_host(void* dst_host, const void* src_device, size_t bytes, bool blocking) {
    if (bytes == 0 || !dst_host || !src_device) return;
    auto event = queue_.memcpy(dst_host, src_device, bytes);
    if (blocking) {
        event.wait();
    }
}

void DeviceContext::copy_device_to_device(void* dst_device, const void* src_device, size_t bytes, bool blocking) {
    if (bytes == 0 || !dst_device || !src_device) return;
    auto event = queue_.memcpy(dst_device, src_device, bytes);
    if (blocking) {
        event.wait();
    }
}

void DeviceContext::fill_device(void* dst_device, uint8_t value, size_t bytes, bool blocking) {
    if (bytes == 0 || !dst_device) return;
    auto event = queue_.memset(dst_device, value, bytes);
    if (blocking) {
        event.wait();
    }
}

void DeviceContext::synchronize() {
    queue_.wait();
}

// ----------------------------------------------------------------------------
// Image dimension parsing and validation (fail-loud contract)
// ----------------------------------------------------------------------------

bool parse_image_dimensions(const uint8_t* data, size_t size, ImageDimensions& out_dims, std::string* error_msg) {
    if (!data || size == 0) {
        if (error_msg) *error_msg = "Image data buffer is empty";
        return false;
    }

    // 1. Check PNG: 8-byte signature 89 50 4E 47 0D 0A 1A 0A
    if (size >= 24 &&
        data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E && data[3] == 0x47 &&
        data[4] == 0x0D && data[5] == 0x0A && data[6] == 0x1A && data[7] == 0x0A) {
        // IHDR chunk: starts at byte 12 ("IHDR")
        if (data[12] == 'I' && data[13] == 'H' && data[14] == 'D' && data[15] == 'R') {
            uint32_t w = (static_cast<uint32_t>(data[16]) << 24) |
                         (static_cast<uint32_t>(data[17]) << 16) |
                         (static_cast<uint32_t>(data[18]) << 8)  |
                         static_cast<uint32_t>(data[19]);
            uint32_t h = (static_cast<uint32_t>(data[20]) << 24) |
                         (static_cast<uint32_t>(data[21]) << 16) |
                         (static_cast<uint32_t>(data[22]) << 8)  |
                         static_cast<uint32_t>(data[23]);
            out_dims.width = static_cast<int64_t>(w);
            out_dims.height = static_cast<int64_t>(h);
            out_dims.channels = 3;
            return true;
        }
    }

    // 2. Check BMP: 'B' 'M' signature
    if (size >= 26 && data[0] == 'B' && data[1] == 'M') {
        int32_t w = static_cast<int32_t>(data[18] | (data[19] << 8) | (data[20] << 16) | (data[21] << 24));
        int32_t h = static_cast<int32_t>(data[22] | (data[23] << 8) | (data[24] << 16) | (data[25] << 24));
        out_dims.width = static_cast<int64_t>(std::abs(w));
        out_dims.height = static_cast<int64_t>(std::abs(h));
        out_dims.channels = 3;
        return true;
    }

    // 3. Check JPEG: FF D8 signature
    if (size >= 4 && data[0] == 0xFF && data[1] == 0xD8) {
        size_t pos = 2;
        while (pos + 4 <= size) {
            if (data[pos] != 0xFF) {
                pos++;
                continue;
            }
            uint8_t marker = data[pos + 1];
            if (marker == 0xD9 || marker == 0xDA) { // EOI or SOS
                break;
            }
            if (marker == 0xD8 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {
                pos += 2;
                continue;
            }
            if (pos + 4 > size) break;
            uint16_t seg_len = (static_cast<uint16_t>(data[pos + 2]) << 8) | static_cast<uint16_t>(data[pos + 3]);
            // SOF0..SOF15 (except DHT 0xC4, JPG 0xC8, DAC 0xCC)
            if (marker >= 0xC0 && marker <= 0xCF && marker != 0xC4 && marker != 0xC8 && marker != 0xCC) {
                if (pos + 9 <= size) {
                    uint16_t h = (static_cast<uint16_t>(data[pos + 5]) << 8) | static_cast<uint16_t>(data[pos + 6]);
                    uint16_t w = (static_cast<uint16_t>(data[pos + 7]) << 8) | static_cast<uint16_t>(data[pos + 8]);
                    uint8_t comps = data[pos + 9];
                    out_dims.width = static_cast<int64_t>(w);
                    out_dims.height = static_cast<int64_t>(h);
                    out_dims.channels = (comps > 0) ? comps : 3;
                    return true;
                }
            }
            pos += 2 + seg_len;
        }
    }

    if (out_dims.width > 0 && out_dims.height > 0) {
        return true;
    }

    if (error_msg) {
        *error_msg = "Unrecognized image format or missing valid image header (supported: PNG, BMP, JPEG)";
    }
    return false;
}

bool validate_image_dimensions(int64_t width, int64_t height,
                               int64_t max_resolution,
                               float max_aspect_ratio,
                               std::string* error_msg) {
    if (width <= 0 || height <= 0) {
        if (error_msg) *error_msg = "Invalid image dimensions: width and height must be positive";
        return false;
    }
    if (width > max_resolution || height > max_resolution) {
        if (error_msg) {
            *error_msg = "Image resolution (" + std::to_string(width) + "x" + std::to_string(height) +
                         ") exceeds maximum allowed resolution (" + std::to_string(max_resolution) + "x" +
                         std::to_string(max_resolution) + ")";
        }
        return false;
    }
    float aspect_ratio = (width >= height) ? static_cast<float>(width) / static_cast<float>(height)
                                           : static_cast<float>(height) / static_cast<float>(width);
    if (aspect_ratio > max_aspect_ratio) {
        if (error_msg) {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(2) << aspect_ratio;
            *error_msg = "Image aspect ratio (" + ss.str() + ":1) exceeds maximum allowed aspect ratio (" +
                         std::to_string(static_cast<int>(max_aspect_ratio)) + ":1)";
        }
        return false;
    }
    return true;
}

// ----------------------------------------------------------------------------
// Offloaded image resizing and normalization to USM
// ----------------------------------------------------------------------------
// Grounding source: docs/vendor/xe-gpu-architecture.md & docs/vendor/thread-mapping-occupancy.md
// Xe2-HPG Battlemage B60 has 64 hardware threads per Xe-Core (8 Vector Engines * 8 threads/VE).
// Launching workgroups of 256 work-items (8 subgroups of 32 work-items) utilizes 8 hardware threads
// per Xe-Core, cleanly mapping to the vector execution pipeline.
// Memory allocations use 64-byte alignment matching the Xe2 cache-line DMA transaction width.

sycl::half* preprocess_image_to_usm(
    DeviceContext& ctx,
    const uint8_t* rgb_pixels,
    int64_t src_w,
    int64_t src_h,
    int64_t channels,
    int64_t dst_w,
    int64_t dst_h,
    bool use_shared_mem,
    sycl::half* out_buffer
) {
    if (!rgb_pixels || src_w <= 0 || src_h <= 0 || channels <= 0) {
        throw std::invalid_argument("Invalid image input buffer or dimensions in preprocess_image_to_usm");
    }

    // Number of patches: 16x16 = 256 patches
    // Patch size: 14x14 pixels
    // In-channels: 3
    // Patch dimension = 3 * 14 * 14 = 588 elements
    // Total half elements = 256 * 588 = 150,528 elements
    const size_t total_elements = 256 * 588; // 150528
    const size_t total_bytes = total_elements * sizeof(sycl::half);

    sycl::half* d_out = out_buffer;
    if (!d_out) {
        if (use_shared_mem) {
            d_out = static_cast<sycl::half*>(sycl::aligned_alloc_shared(64, total_bytes, ctx.queue()));
        } else {
            d_out = static_cast<sycl::half*>(ctx.allocate_device(total_bytes, 64));
        }
    }

    // Allocate temporary device buffer for input RGB image
    size_t src_bytes = static_cast<size_t>(src_w * src_h * channels);
    uint8_t* d_src = static_cast<uint8_t*>(ctx.allocate_device(src_bytes, 64));
    ctx.copy_host_to_device(d_src, rgb_pixels, src_bytes, true);

    // Launch nd_range<1> with work-group size 256. 150,528 / 256 = 588 work-groups.
    auto ev = ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(total_elements), sycl::range<1>(256)),
            [=](sycl::nd_item<1> item) {
                size_t idx = item.get_global_id(0);
                if (idx >= total_elements) return;

                int64_t patch_idx = static_cast<int64_t>(idx / 588);
                int64_t elem_in_patch = static_cast<int64_t>(idx % 588);
                int64_t patch_y = patch_idx / 16;
                int64_t patch_x = patch_idx % 16;

                int64_t c = elem_in_patch / (14 * 14);
                int64_t rem = elem_in_patch % (14 * 14);
                int64_t py = rem / 14;
                int64_t px = rem % 14;

                int64_t cur_dst_y = patch_y * 14 + py;
                int64_t cur_dst_x = patch_x * 14 + px;

                float u = (static_cast<float>(cur_dst_x) + 0.5f) * (static_cast<float>(src_w) / static_cast<float>(dst_w)) - 0.5f;
                float v = (static_cast<float>(cur_dst_y) + 0.5f) * (static_cast<float>(src_h) / static_cast<float>(dst_h)) - 0.5f;

                int64_t x0 = sycl::clamp(static_cast<int64_t>(sycl::floor(u)), (int64_t)0, src_w - 1);
                int64_t x1 = sycl::clamp(x0 + 1, (int64_t)0, src_w - 1);
                int64_t y0 = sycl::clamp(static_cast<int64_t>(sycl::floor(v)), (int64_t)0, src_h - 1);
                int64_t y1 = sycl::clamp(y0 + 1, (int64_t)0, src_h - 1);

                float fx = u - sycl::floor(u);
                float fy = v - sycl::floor(v);
                if (x0 == x1) fx = 0.0f;
                if (y0 == y1) fy = 0.0f;

                float p00 = static_cast<float>(d_src[(y0 * src_w + x0) * channels + c]);
                float p10 = static_cast<float>(d_src[(y0 * src_w + x1) * channels + c]);
                float p01 = static_cast<float>(d_src[(y1 * src_w + x0) * channels + c]);
                float p11 = static_cast<float>(d_src[(y1 * src_w + x1) * channels + c]);

                float sample = (p00 * (1.0f - fx) + p10 * fx) * (1.0f - fy) +
                               (p01 * (1.0f - fx) + p11 * fx) * fy;

                // Normalize [0, 255] -> [-1.0, 1.0]
                float norm_val = (sample / 127.5f) - 1.0f;
                d_out[idx] = static_cast<sycl::half>(norm_val);
            });
    });
    ev.wait();
    ctx.free_device(d_src);
    return d_out;
}

} // namespace xinfer::core

