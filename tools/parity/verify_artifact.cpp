#include "artifact/reader.h"
#include <iostream>
#include <iomanip>
#include <chrono>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: verify_artifact <path_to_artifact.xinfer> [--check-all-sections]" << std::endl;
        return 1;
    }

    std::string artifact_path = argv[1];
    bool check_all_sections = (argc >= 3 && std::string(argv[2]) == "--check-all-sections");

    std::cout << "===========================================" << std::endl;
    std::cout << " xinfer Artifact Verification Tool" << std::endl;
    std::cout << " File: " << artifact_path << std::endl;
    std::cout << "===========================================" << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();

    xinfer::artifact::ArtifactReader reader;
    std::string err;
    if (!reader.open(artifact_path, &err)) {
        std::cerr << "FAILED to open artifact: " << err << std::endl;
        return 1;
    }

    const auto& header = reader.header();
    const auto& meta = reader.metadata();

    std::cout << "\n[1/3] Header & Metadata Validation:" << std::endl;
    std::cout << "  Format Version:   " << header.format_version << std::endl;
    std::cout << "  Section Count:    " << header.section_count << std::endl;
    std::cout << "  Total File Size:  " << std::fixed << std::setprecision(2)
              << (header.total_file_size / (1024.0 * 1024.0 * 1024.0)) << " GB ("
              << header.total_file_size << " bytes)" << std::endl;
    std::cout << "  Model Name:       " << meta.model_name << std::endl;
    std::cout << "  Quant Scheme:     " << meta.quant_scheme << std::endl;
    std::cout << "  Tokenizer Type:   " << meta.tokenizer_type << std::endl;
    std::cout << "  Arch Properties:  " << meta.properties.size() << " parameters" << std::endl;

    std::cout << "\n[2/3] Validating Full File Checksum (CRC-64/ECMA-182)..." << std::endl;
    auto t_crc_start = std::chrono::high_resolution_clock::now();
    if (!reader.validate_checksum(&err)) {
        std::cerr << "FAILED whole-file checksum verification: " << err << std::endl;
        return 1;
    }
    auto t_crc_end = std::chrono::high_resolution_clock::now();
    double crc_sec = std::chrono::duration<double>(t_crc_end - t_crc_start).count();
    double mb_per_sec = (header.total_file_size / (1024.0 * 1024.0)) / (crc_sec > 0 ? crc_sec : 1e-6);

    std::cout << "  CRC-64 Digest:    0x" << std::hex << reader.footer().file_checksum << std::dec << std::endl;
    std::cout << "  Checksum Verified in " << std::fixed << std::setprecision(2) << crc_sec << " s ("
              << mb_per_sec << " MB/s)" << std::endl;

    std::cout << "\n[3/3] Inspecting Section Table Layout..." << std::endl;
    auto section_names = reader.list_sections();
    size_t aligned_count = 0;
    for (const auto& name : section_names) {
        auto info = reader.get_section_info(name);
        if (info && (info->offset % xinfer::artifact::SECTION_ALIGNMENT == 0)) {
            aligned_count++;
        }
    }
    std::cout << "  Sections: " << section_names.size() << " total, "
              << aligned_count << " 64-byte aligned." << std::endl;

    if (aligned_count != section_names.size()) {
        std::cerr << "WARNING: " << (section_names.size() - aligned_count)
                  << " sections are not 64-byte aligned!" << std::endl;
        return 1;
    }

    if (check_all_sections) {
        std::cout << "\n  Validating all section checksums individually..." << std::endl;
        std::vector<uint8_t> buffer;
        for (size_t i = 0; i < section_names.size(); ++i) {
            const auto& name = section_names[i];
            if (!reader.read_section(name, buffer, &err)) {
                std::cerr << "FAILED on section " << name << ": " << err << std::endl;
                return 1;
            }
            if ((i + 1) % 100 == 0 || i + 1 == section_names.size()) {
                std::cout << "  Verified " << (i + 1) << "/" << section_names.size() << " sections...\r" << std::flush;
            }
        }
        std::cout << std::endl;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\n===========================================" << std::endl;
    std::cout << " STATUS: ARTIFACT VALIDATION PASSED!" << std::endl;
    std::cout << " Total Time: " << std::fixed << std::setprecision(2) << total_sec << " s" << std::endl;
    std::cout << "===========================================" << std::endl;

    return 0;
}
