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

# Milestone 1: Round-trip test
add_executable(test_artifact_roundtrip
    tests/test_artifact_roundtrip.cpp
)

target_link_libraries(test_artifact_roundtrip PRIVATE
    xinfer_artifact
)

enable_testing()
add_test(NAME ArtifactRoundTrip COMMAND test_artifact_roundtrip)
