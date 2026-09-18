#include "xinfer/engine.h"
#include "artifact/writer.h"
#include "artifact/reader.h"
#include <iostream>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;
using namespace xinfer;
using namespace xinfer::artifact;

int main() {
    std::cout << "=== Running test_engine_checksum ===" << std::endl;

    const std::string valid_filepath = "test_chk_valid.xinfer";
    const std::string corrupt_filepath = "test_chk_corrupt.xinfer";

    // Clean up any leftovers
    if (fs::exists(valid_filepath)) fs::remove(valid_filepath);
    if (fs::exists(corrupt_filepath)) fs::remove(corrupt_filepath);

    // 1. Create a minimal valid artifact
    {
        ArtifactMetadata meta;
        meta.model_name = "Qwen/Qwen3.8-27B";
        meta.quant_scheme = "INT4-G128-SYM";
        meta.tokenizer_type = "qwen3_8_tiktoken";

        std::vector<uint8_t> dummy_payload(1024, 0xAB);

        ArtifactWriter writer;
        writer.set_metadata(meta);
        assert(writer.add_section("test.section", SectionType::RawBlob, dummy_payload));

        std::string err;
        bool ok = writer.write_to_file(valid_filepath, &err);
        assert(ok);
    }

    // 2. Create a corrupted copy by modifying 1 byte in the payload without changing file size
    {
        std::ifstream src(valid_filepath, std::ios::binary);
        std::vector<char> content((std::istreambuf_iterator<char>(src)),
                                   std::istreambuf_iterator<char>());
        src.close();

        // Flip a byte in the payload (offset 100)
        assert(content.size() > 150);
        content[100] ^= 0xFF;

        std::ofstream dst(corrupt_filepath, std::ios::binary);
        dst.write(content.data(), content.size());
        dst.close();
    }

    // 3. Confirm ArtifactReader::open() succeeds on the corrupted file because file size matches
    {
        ArtifactReader reader;
        std::string err;
        bool open_ok = reader.open(corrupt_filepath, &err);
        assert(open_ok);
        std::cout << "1. ArtifactReader::open() succeeded on corrupted file as expected (file size matches header)." << std::endl;

        // But validate_checksum() must detect the mismatch!
        bool chk_ok = reader.validate_checksum(&err);
        assert(!chk_ok);
        std::cout << "2. ArtifactReader::validate_checksum() caught corruption: " << err << std::endl;
    }

    // 4. Test Engine::load with validate_checksum = true on corrupted file
    {
        Engine engine;
        EngineConfig cfg;
        cfg.artifact_path = corrupt_filepath;
        cfg.validate_checksum = true;

        std::string err;
        bool loaded = engine.load(cfg, &err);
        assert(!loaded);
        assert(err.find("checksum") != std::string::npos);
        std::cout << "3. Engine::load() with validate_checksum=true rejected corrupted artifact: " << err << std::endl;
    }

    // 5. Test Engine::load with validate_checksum = false on corrupted file
    // Checksum check is bypassed; load fails later due to missing model weights/tokenizer, NOT checksum
    {
        Engine engine;
        EngineConfig cfg;
        cfg.artifact_path = corrupt_filepath;
        cfg.validate_checksum = false;

        std::string err;
        bool loaded = engine.load(cfg, &err);
        assert(!loaded);
        assert(err.find("checksum") == std::string::npos);
        std::cout << "4. Engine::load() with validate_checksum=false bypassed checksum verification (failed as expected later on: " << err << ")" << std::endl;
    }

    // Cleanup
    fs::remove(valid_filepath);
    fs::remove(corrupt_filepath);

    std::cout << "=== test_engine_checksum passed successfully! ===" << std::endl;
    return 0;
}
