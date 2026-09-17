cmake_minimum_required(VERSION 3.20)
project(xinfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Common compiler flags for MSVC-compatible compilers (icx-cl / cl)
if (MSVC)
    add_compile_options(/EHsc /W4)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# Core artifact library (CPU-only, no SYCL/GPU dependency)
add_library(xinfer_artifact STATIC
    src/artifact/checksum.cpp
    src/artifact/metadata.cpp
    src/artifact/writer.cpp
    src/artifact/reader.cpp
)

target_include_directories(xinfer_artifact PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# C API shared library for fast Python converter integration
add_library(xinfer_artifact_c SHARED
    src/artifact/checksum.cpp
    src/artifact/c_api.cpp
)
target_include_directories(xinfer_artifact_c PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Milestone 1: Round-trip test
add_executable(test_artifact_roundtrip
    tests/test_artifact_roundtrip.cpp
)
target_link_libraries(test_artifact_roundtrip PRIVATE
    xinfer_artifact
)

# Milestone 2: Artifact inspection and parity verification tool
add_executable(verify_artifact
    tools/parity/verify_artifact.cpp
)
target_link_libraries(verify_artifact PRIVATE
    xinfer_artifact
)

# Milestone 3: Core device layer (SYCL / Level Zero primitives)
add_library(xinfer_core STATIC
    src/core/device.cpp
    src/core/tensor.cpp
    src/core/arena.cpp
    src/core/command_list.cpp
    src/core/kv_cache.cpp
)
target_include_directories(xinfer_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
if (UNIX AND NOT APPLE)
    target_link_libraries(xinfer_core PUBLIC ${CMAKE_DL_LIBS})
endif()
target_compile_options(xinfer_core PUBLIC -fsycl)
target_link_options(xinfer_core PUBLIC -fsycl)

add_executable(test_core_device
    tests/test_core_device.cpp
)
target_link_libraries(test_core_device PRIVATE
    xinfer_core
)
target_compile_options(test_core_device PRIVATE -fsycl)
target_link_options(test_core_device PRIVATE -fsycl)

add_executable(test_kv_cache
    tests/test_kv_cache.cpp
)
target_link_libraries(test_kv_cache PRIVATE
    xinfer_core
    xinfer_ops
)
target_compile_options(test_kv_cache PRIVATE -fsycl)
target_link_options(test_kv_cache PRIVATE -fsycl)

enable_testing()
add_test(NAME ArtifactRoundTrip COMMAND test_artifact_roundtrip)
add_test(NAME CoreDeviceTest COMMAND test_core_device)
add_test(NAME KVCacheTest COMMAND test_kv_cache)

# Milestone 4: Naive correctness kernels (no XMX yet)
add_library(xinfer_ops STATIC
    src/ops/rmsnorm.cpp
    src/ops/rope.cpp
    src/ops/elementwise.cpp
    src/ops/softmax.cpp
    src/ops/linear.cpp
    src/ops/gemm_xmx.cpp
    src/ops/attention.cpp
    src/ops/sampling.cpp
)
target_include_directories(xinfer_ops PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(xinfer_ops PUBLIC
    xinfer_core
)
target_compile_options(xinfer_ops PUBLIC -fsycl)
target_link_options(xinfer_ops PUBLIC -fsycl)

add_executable(test_ops_oracle
    tests/test_ops_oracle.cpp
)
target_link_libraries(test_ops_oracle PRIVATE
    xinfer_ops
    xinfer_core
)
target_compile_options(test_ops_oracle PRIVATE -fsycl)
target_link_options(test_ops_oracle PRIVATE -fsycl)

add_test(NAME OpsOracleTest COMMAND test_ops_oracle)

# Milestone 5: Minimal Engine (End-to-End Single-Request Decode)
add_library(xinfer_targets STATIC
    src/targets/qwen3_8/chat_template.cpp
    src/targets/qwen3_8/tokenizer.cpp
    src/targets/qwen3_8/linear_attn.cpp
    src/targets/qwen3_8/forward.cpp
    src/targets/qwen3_8/decode_graph.cpp
    src/targets/qwen3_8_27b/weights.cpp
)
target_include_directories(xinfer_targets PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(xinfer_targets PUBLIC
    xinfer_ops
    xinfer_core
    xinfer_artifact
)
target_compile_options(xinfer_targets PUBLIC -fsycl)
target_link_options(xinfer_targets PUBLIC -fsycl)

add_library(xinfer_runtime STATIC
    src/runtime/engine.cpp
)
target_include_directories(xinfer_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(xinfer_runtime PUBLIC
    xinfer_targets
)
target_compile_options(xinfer_runtime PUBLIC -fsycl)
target_link_options(xinfer_runtime PUBLIC -fsycl)

add_executable(xinfer
    apps/xinfer/main.cpp
)
target_include_directories(xinfer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(xinfer PRIVATE
    xinfer_runtime
)
target_compile_options(xinfer PRIVATE -fsycl)
target_link_options(xinfer PRIVATE -fsycl)

add_executable(test_decode_graph
    tests/test_decode_graph.cpp
)
target_link_libraries(test_decode_graph PRIVATE
    xinfer_targets
    xinfer_runtime
)
target_compile_options(test_decode_graph PRIVATE -fsycl)
target_link_options(test_decode_graph PRIVATE -fsycl)

add_test(NAME DecodeGraphTest COMMAND test_decode_graph)

add_executable(test_tokenizer_bpe
    tests/test_tokenizer_bpe.cpp
)
target_include_directories(test_tokenizer_bpe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_tokenizer_bpe PRIVATE
    xinfer_targets
    xinfer_runtime
)
target_compile_options(test_tokenizer_bpe PRIVATE -fsycl)
target_link_options(test_tokenizer_bpe PRIVATE -fsycl)

add_test(NAME TokenizerBPETest COMMAND test_tokenizer_bpe)

add_executable(test_chat_template
    tests/test_chat_template.cpp
)
target_include_directories(test_chat_template PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_chat_template PRIVATE
    xinfer_targets
    xinfer_runtime
)
target_compile_options(test_chat_template PRIVATE -fsycl)
target_link_options(test_chat_template PRIVATE -fsycl)

add_test(NAME ChatTemplateTest COMMAND test_chat_template)

# Milestone 9: OpenAI-Compatible HTTP Serving
add_library(xinfer_serve STATIC
    src/serve/protocol.cpp
    src/serve/server.cpp
)
target_include_directories(xinfer_serve PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(xinfer_serve PUBLIC
    xinfer_runtime
)
if (WIN32)
    target_link_libraries(xinfer_serve PUBLIC ws2_32)
endif()
target_compile_options(xinfer_serve PUBLIC -fsycl)
target_link_options(xinfer_serve PUBLIC -fsycl)

add_executable(xinfer-serve
    apps/xinfer-serve/main.cpp
)
target_include_directories(xinfer-serve PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(xinfer-serve PRIVATE
    xinfer_serve
)
target_compile_options(xinfer-serve PRIVATE -fsycl)
target_link_options(xinfer-serve PRIVATE -fsycl)

add_executable(test_serve_schema
    tests/test_serve_schema.cpp
)
target_link_libraries(test_serve_schema PRIVATE
    xinfer_serve
)
target_compile_options(test_serve_schema PRIVATE -fsycl)
target_link_options(test_serve_schema PRIVATE -fsycl)

add_test(NAME ServeSchemaTest COMMAND test_serve_schema)

# Milestone 10: Performance Diagnostic & Profiler
add_executable(profile_decode_step
    tools/parity/profile_decode_step.cpp
)
target_include_directories(profile_decode_step PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(profile_decode_step PRIVATE
    xinfer_targets
    xinfer_runtime
)
target_compile_options(profile_decode_step PRIVATE -fsycl)
target_link_options(profile_decode_step PRIVATE -fsycl)

add_executable(profile_projections
    tools/parity/profile_projections.cpp
)
target_include_directories(profile_projections PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(profile_projections PRIVATE
    xinfer_targets
    xinfer_runtime
)
target_compile_options(profile_projections PRIVATE -fsycl)
target_link_options(profile_projections PRIVATE -fsycl)

add_executable(verify_pcie_and_memory
    tools/parity/verify_pcie_and_memory.cpp
)
target_include_directories(verify_pcie_and_memory PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(verify_pcie_and_memory PRIVATE
    xinfer_targets
    xinfer_runtime
)
target_compile_options(verify_pcie_and_memory PRIVATE -fsycl)
target_link_options(verify_pcie_and_memory PRIVATE -fsycl)

add_executable(dump_engine_logits
    tools/parity/dump_engine_logits.cpp
)
target_include_directories(dump_engine_logits PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(dump_engine_logits PRIVATE
    xinfer_targets
    xinfer_runtime
)
target_compile_options(dump_engine_logits PRIVATE -fsycl)
target_link_options(dump_engine_logits PRIVATE -fsycl)
