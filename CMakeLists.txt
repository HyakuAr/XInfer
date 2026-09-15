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
)
target_include_directories(xinfer_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
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

enable_testing()
add_test(NAME ArtifactRoundTrip COMMAND test_artifact_roundtrip)
add_test(NAME CoreDeviceTest COMMAND test_core_device)

# Milestone 4: Naive correctness kernels (no XMX yet)
add_library(xinfer_ops STATIC
    src/ops/rmsnorm.cpp
    src/ops/rope.cpp
    src/ops/elementwise.cpp
    src/ops/softmax.cpp
    src/ops/linear.cpp
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
