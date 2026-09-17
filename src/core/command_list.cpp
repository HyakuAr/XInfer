#include "command_list.h"
#include <stdexcept>
#include <iostream>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// Grounding source: docs/vendor/level-zero-command-lists.md
// Level Zero Specification (v1.18+):
// - Regular deferred command lists (record once, replay many times)
// - Descriptors: ze_command_list_desc_t, ze_command_queue_desc_t, ze_fence_desc_t
#if __has_include(<level_zero/ze_api.h>)
#include <level_zero/ze_api.h>
#else
namespace {

typedef uint32_t ze_result_t;
typedef uint32_t ze_structure_type_t;

constexpr ze_result_t ZE_RESULT_SUCCESS = 0;
constexpr ze_structure_type_t ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC = 0x10002;
constexpr ze_structure_type_t ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC = 0x10003;
constexpr ze_structure_type_t ZE_STRUCTURE_TYPE_FENCE_DESC = 0x10006;

#pragma pack(push, 1)
struct ze_command_list_desc_t {
    ze_structure_type_t stype;
    const void* pNext;
    uint32_t commandQueueGroupOrdinal;
    uint32_t flags;
};

struct ze_command_queue_desc_t {
    ze_structure_type_t stype;
    const void* pNext;
    uint32_t ordinal;
    uint32_t index;
    uint32_t flags;
    uint32_t mode;
    uint32_t priority;
};

struct ze_fence_desc_t {
    ze_structure_type_t stype;
    const void* pNext;
    uint32_t flags;
};
#pragma pack(pop)

} // anonymous namespace
#endif

namespace xinfer::core {

struct LevelZeroCommandList::L0DispatchTable {
    ze_result_t (*pfnCommandListCreate)(void*, void*, const ze_command_list_desc_t*, void**){nullptr};
    ze_result_t (*pfnCommandListClose)(void*){nullptr};
    ze_result_t (*pfnCommandListReset)(void*){nullptr};
    ze_result_t (*pfnCommandListDestroy)(void*){nullptr};
    ze_result_t (*pfnCommandListAppendMemoryCopy)(void*, void*, const void*, size_t, void*, uint32_t, void**){nullptr};
    ze_result_t (*pfnCommandListAppendBarrier)(void*, void*, uint32_t, void**){nullptr};
    ze_result_t (*pfnCommandQueueCreate)(void*, void*, const ze_command_queue_desc_t*, void**){nullptr};
    ze_result_t (*pfnCommandQueueDestroy)(void*){nullptr};
    ze_result_t (*pfnCommandQueueExecuteCommandLists)(void*, uint32_t, void**, void*){nullptr};
    ze_result_t (*pfnCommandQueueSynchronize)(void*, uint64_t){nullptr};
    ze_result_t (*pfnFenceCreate)(void*, const ze_fence_desc_t*, void**){nullptr};
    ze_result_t (*pfnFenceDestroy)(void*){nullptr};
    ze_result_t (*pfnFenceHostSynchronize)(void*, uint64_t){nullptr};
    ze_result_t (*pfnFenceReset)(void*){nullptr};
};

std::unique_ptr<LevelZeroCommandList> LevelZeroCommandList::create(std::shared_ptr<DeviceContext> ctx) {
    auto cl = std::unique_ptr<LevelZeroCommandList>(new LevelZeroCommandList(std::move(ctx)));
    cl->init();
    return cl;
}

LevelZeroCommandList::LevelZeroCommandList(std::shared_ptr<DeviceContext> ctx)
    : ctx_(std::move(ctx)), fn_(std::make_unique<L0DispatchTable>()) {}

LevelZeroCommandList::~LevelZeroCommandList() {
    if (fn_) {
        if (fence_ && fn_->pfnFenceDestroy) fn_->pfnFenceDestroy(fence_);
        if (cmd_list_ && fn_->pfnCommandListDestroy) fn_->pfnCommandListDestroy(cmd_list_);
        if (cmd_queue_ && fn_->pfnCommandQueueDestroy) fn_->pfnCommandQueueDestroy(cmd_queue_);
    }
    if (h_module_) {
#ifdef _WIN32
        FreeLibrary(static_cast<HMODULE>(h_module_));
#else
        dlclose(h_module_);
#endif
        h_module_ = nullptr;
    }
}

LevelZeroCommandList::LevelZeroCommandList(LevelZeroCommandList&& other) noexcept
    : ctx_(std::move(other.ctx_)),
      h_module_(other.h_module_),
      cmd_list_(other.cmd_list_),
      cmd_queue_(other.cmd_queue_),
      fence_(other.fence_),
      is_closed_(other.is_closed_),
      fn_(std::move(other.fn_)) {
    other.h_module_ = nullptr;
    other.cmd_list_ = nullptr;
    other.cmd_queue_ = nullptr;
    other.fence_ = nullptr;
    other.is_closed_ = false;
}

LevelZeroCommandList& LevelZeroCommandList::operator=(LevelZeroCommandList&& other) noexcept {
    if (this != &other) {
        if (fn_) {
            if (fence_ && fn_->pfnFenceDestroy) fn_->pfnFenceDestroy(fence_);
            if (cmd_list_ && fn_->pfnCommandListDestroy) fn_->pfnCommandListDestroy(cmd_list_);
            if (cmd_queue_ && fn_->pfnCommandQueueDestroy) fn_->pfnCommandQueueDestroy(cmd_queue_);
        }
        if (h_module_) {
#ifdef _WIN32
            FreeLibrary(static_cast<HMODULE>(h_module_));
#else
            dlclose(h_module_);
#endif
            h_module_ = nullptr;
        }

        ctx_ = std::move(other.ctx_);
        h_module_ = other.h_module_;
        cmd_list_ = other.cmd_list_;
        cmd_queue_ = other.cmd_queue_;
        fence_ = other.fence_;
        is_closed_ = other.is_closed_;
        fn_ = std::move(other.fn_);

        other.h_module_ = nullptr;
        other.cmd_list_ = nullptr;
        other.cmd_queue_ = nullptr;
        other.fence_ = nullptr;
        other.is_closed_ = false;
    }
    return *this;
}

void LevelZeroCommandList::init() {
    if (!ctx_ || !ctx_->has_native_level_zero()) {
        throw std::runtime_error("LevelZeroCommandList requires a device context with Level Zero backend");
    }

#ifdef _WIN32
    HMODULE hZe = LoadLibraryA("ze_loader.dll");
    if (!hZe) {
        throw std::runtime_error("Failed to load ze_loader.dll from system (Level Zero driver/loader missing)");
    }
    h_module_ = static_cast<void*>(hZe);
    auto load_sym = [hZe](const char* sym_name) -> void* {
        return reinterpret_cast<void*>(GetProcAddress(hZe, sym_name));
    };
#else
    // POSIX dynamic loading for Linux Level Zero loader (libze_loader.so.1 / libze_loader.so)
    void* hZe = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!hZe) {
        hZe = dlopen("libze_loader.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (!hZe) {
        const char* err = dlerror();
        throw std::runtime_error(std::string("Failed to load libze_loader.so from system: ") + (err ? err : "unknown error"));
    }
    h_module_ = hZe;
    auto load_sym = [hZe](const char* sym_name) -> void* {
        return dlsym(hZe, sym_name);
    };
#endif

#define LOAD_ZE_FN(fn_name, member) \
    do { \
        void* p = load_sym(fn_name); \
        if (!p) { \
            throw std::runtime_error(std::string("Failed to resolve Level Zero symbol: ") + (fn_name)); \
        } \
        fn_->member = reinterpret_cast<decltype(fn_->member)>(p); \
    } while (0)

    LOAD_ZE_FN("zeCommandListCreate", pfnCommandListCreate);
    LOAD_ZE_FN("zeCommandListClose", pfnCommandListClose);
    LOAD_ZE_FN("zeCommandListReset", pfnCommandListReset);
    LOAD_ZE_FN("zeCommandListDestroy", pfnCommandListDestroy);
    LOAD_ZE_FN("zeCommandListAppendMemoryCopy", pfnCommandListAppendMemoryCopy);
    LOAD_ZE_FN("zeCommandListAppendBarrier", pfnCommandListAppendBarrier);
    LOAD_ZE_FN("zeCommandQueueCreate", pfnCommandQueueCreate);
    LOAD_ZE_FN("zeCommandQueueDestroy", pfnCommandQueueDestroy);
    LOAD_ZE_FN("zeCommandQueueExecuteCommandLists", pfnCommandQueueExecuteCommandLists);
    LOAD_ZE_FN("zeCommandQueueSynchronize", pfnCommandQueueSynchronize);
    LOAD_ZE_FN("zeFenceCreate", pfnFenceCreate);
    LOAD_ZE_FN("zeFenceDestroy", pfnFenceDestroy);
    LOAD_ZE_FN("zeFenceHostSynchronize", pfnFenceHostSynchronize);
    LOAD_ZE_FN("zeFenceReset", pfnFenceReset);
#undef LOAD_ZE_FN

    void* hDevice = ctx_->native_l0_device();
    void* hContext = ctx_->native_l0_context();

    // Create regular deferred command list (record once, replay many times per docs/vendor/level-zero-command-lists.md)
    ze_command_list_desc_t clDesc{};
    clDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    clDesc.commandQueueGroupOrdinal = 0;
    clDesc.flags = 0;

    ze_result_t res = fn_->pfnCommandListCreate(hContext, hDevice, &clDesc, &cmd_list_);
    if (res != ZE_RESULT_SUCCESS || !cmd_list_) {
        throw std::runtime_error("zeCommandListCreate failed with error code: " + std::to_string(res));
    }

    // Create execution command queue
    ze_command_queue_desc_t cqDesc{};
    cqDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    cqDesc.ordinal = 0;
    cqDesc.index = 0;
    cqDesc.flags = 0;
    cqDesc.mode = 0;
    cqDesc.priority = 0;

    res = fn_->pfnCommandQueueCreate(hContext, hDevice, &cqDesc, &cmd_queue_);
    if (res != ZE_RESULT_SUCCESS || !cmd_queue_) {
        throw std::runtime_error("zeCommandQueueCreate failed with error code: " + std::to_string(res));
    }

    // Create synchronization fence
    ze_fence_desc_t fDesc{};
    fDesc.stype = ZE_STRUCTURE_TYPE_FENCE_DESC;
    fDesc.flags = 0;

    res = fn_->pfnFenceCreate(cmd_queue_, &fDesc, &fence_);
    if (res != ZE_RESULT_SUCCESS || !fence_) {
        throw std::runtime_error("zeFenceCreate failed with error code: " + std::to_string(res));
    }

    is_closed_ = false;
}

void LevelZeroCommandList::append_memory_copy(void* dst_device, const void* src_device, size_t bytes) {
    if (is_closed_) {
        throw std::runtime_error("Cannot append commands to a closed LevelZeroCommandList");
    }
    ze_result_t res = fn_->pfnCommandListAppendMemoryCopy(cmd_list_, dst_device, src_device, bytes, nullptr, 0, nullptr);
    if (res != ZE_RESULT_SUCCESS) {
        throw std::runtime_error("zeCommandListAppendMemoryCopy failed with error code: " + std::to_string(res));
    }
}

void LevelZeroCommandList::append_barrier() {
    if (is_closed_) {
        throw std::runtime_error("Cannot append barrier to a closed LevelZeroCommandList");
    }
    ze_result_t res = fn_->pfnCommandListAppendBarrier(cmd_list_, nullptr, 0, nullptr);
    if (res != ZE_RESULT_SUCCESS) {
        throw std::runtime_error("zeCommandListAppendBarrier failed with error code: " + std::to_string(res));
    }
}

void LevelZeroCommandList::close() {
    if (!is_closed_) {
        ze_result_t res = fn_->pfnCommandListClose(cmd_list_);
        if (res != ZE_RESULT_SUCCESS) {
            throw std::runtime_error("zeCommandListClose failed with error code: " + std::to_string(res));
        }
        is_closed_ = true;
    }
}

void LevelZeroCommandList::reset() {
    if (is_closed_) {
        ze_result_t res = fn_->pfnCommandListReset(cmd_list_);
        if (res != ZE_RESULT_SUCCESS) {
            throw std::runtime_error("zeCommandListReset failed with error code: " + std::to_string(res));
        }
        is_closed_ = false;
    }
}

void LevelZeroCommandList::execute() {
    if (!is_closed_) {
        close();
    }
    fn_->pfnFenceReset(fence_);
    ze_result_t res = fn_->pfnCommandQueueExecuteCommandLists(cmd_queue_, 1, &cmd_list_, fence_);
    if (res != ZE_RESULT_SUCCESS) {
        throw std::runtime_error("zeCommandQueueExecuteCommandLists failed with error code: " + std::to_string(res));
    }
}

void LevelZeroCommandList::synchronize() {
    ze_result_t res = fn_->pfnFenceHostSynchronize(fence_, UINT64_MAX);
    if (res != ZE_RESULT_SUCCESS) {
        throw std::runtime_error("zeFenceHostSynchronize failed with error code: " + std::to_string(res));
    }
}

} // namespace xinfer::core
