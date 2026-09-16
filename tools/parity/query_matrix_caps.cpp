#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <iostream>
#include <iomanip>

using namespace sycl::ext::oneapi::experimental::matrix;

const char* to_string(matrix_type t) {
    switch (t) {
        case matrix_type::bf16: return "bf16";
        case matrix_type::fp16: return "fp16";
        case matrix_type::tf32: return "tf32";
        case matrix_type::fp32: return "fp32";
        case matrix_type::fp64: return "fp64";
        case matrix_type::sint8: return "sint8";
        case matrix_type::sint16: return "sint16";
        case matrix_type::sint32: return "sint32";
        case matrix_type::sint64: return "sint64";
        case matrix_type::uint8: return "uint8";
        case matrix_type::uint16: return "uint16";
        case matrix_type::uint32: return "uint32";
        case matrix_type::uint64: return "uint64";
        default: return "unknown";
    }
}

int main() {
    sycl::device dev{sycl::gpu_selector_v};
    std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Has ext_intel_matrix: " << std::boolalpha << dev.has(sycl::aspect::ext_intel_matrix) << std::endl;

    if (!dev.has(sycl::aspect::ext_intel_matrix)) {
        std::cout << "Device does not support ext_intel_matrix aspect!" << std::endl;
        return 1;
    }

    auto combinations = dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
    std::cout << "Total supported matrix combinations: " << combinations.size() << std::endl;

    std::cout << std::left 
              << std::setw(8) << "atype"
              << std::setw(8) << "btype"
              << std::setw(8) << "ctype"
              << std::setw(8) << "dtype"
              << std::setw(8) << "max_m"
              << std::setw(8) << "max_n"
              << std::setw(8) << "max_k"
              << std::setw(8) << "msize"
              << std::setw(8) << "nsize"
              << std::setw(8) << "ksize"
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (const auto& c : combinations) {
        std::cout << std::left
                  << std::setw(8) << to_string(c.atype)
                  << std::setw(8) << to_string(c.btype)
                  << std::setw(8) << to_string(c.ctype)
                  << std::setw(8) << to_string(c.dtype)
                  << std::setw(8) << c.max_msize
                  << std::setw(8) << c.max_nsize
                  << std::setw(8) << c.max_ksize
                  << std::setw(8) << c.msize
                  << std::setw(8) << c.nsize
                  << std::setw(8) << c.ksize
                  << std::endl;
    }

    return 0;
}
