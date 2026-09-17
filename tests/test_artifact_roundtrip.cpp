#include "artifact/container.h"
#include "artifact/writer.h"
#include "artifact/reader.h"
#include "artifact/checksum.h"

#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <filesystem>
#include <cstring>

namespace fs = std::filesystem;
using namespace xinfer::artifact;

static std::vector<uint8_t> generate_dummy_bytes(size_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<uint8_t>(dist(rng));
    }
    return data;
}

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << " Running M1 Artifact Format Round-Trip Test" << std::endl;
    std::cout << "===========================================" << std::endl;

    const std::string test_filepath = "test_dummy.xinfer";
    if (fs::exists(test_filepath)) {
        fs::remove(test_filepath);
    }

    // 1. Prepare dummy metadata
    ArtifactMetadata original_meta;
    original_meta.model_name = "Qwen/Qwen3.8-27B";
    original_meta.quant_scheme = "INT4-G128-SYM";
    original_meta.tokenizer_type = "qwen3_8_tiktoken";
    original_meta.properties["hidden_size"] = "5120";
    original_meta.properties["num_attention_heads"] = "40";
    original_meta.properties["num_key_value_heads"] = "8";
    original_meta.properties["num_hidden_layers"] = "64";
    original_meta.properties["intermediate_size"] = "27648";

    // 2. Prepare dummy sections with pseudo-random bytes standing in for weights
    auto embed_weights = generate_dummy_bytes(128 * 1024 + 17, 1001); // odd size to test padding
    auto q_proj_weights = generate_dummy_bytes(256 * 1024 + 3, 1002);
    auto q_proj_scales = generate_dummy_bytes(8 * 1024, 1003);
    auto tokenizer_data = generate_dummy_bytes(32 * 1024 + 42, 1004);
    std::string template_str = "{% for message in messages %}{{ message['content'] }}{% endfor %}";
    std::vector<uint8_t> chat_template(template_str.begin(), template_str.end());

    // 3. Write artifact using ArtifactWriter
    std::cout << "[1/5] Writing dummy artifact to: " << test_filepath << " ..." << std::endl;
    {
        ArtifactWriter writer;
        writer.set_metadata(original_meta);

        assert(writer.add_section("model.embed_tokens.weight", SectionType::TensorWeights, embed_weights));
        assert(writer.add_section("model.layers.0.self_attn.q_proj.weight", SectionType::TensorWeights, q_proj_weights));
        assert(writer.add_section("model.layers.0.self_attn.q_proj.scales", SectionType::TensorScales, q_proj_scales));
        assert(writer.add_section("tokenizer.data", SectionType::TokenizerData, tokenizer_data));
        assert(writer.add_section("chat_template.jinja", SectionType::ChatTemplate, chat_template));

        std::string err;
        bool ok = writer.write_to_file(test_filepath, &err);
        if (!ok) {
            std::cerr << "FAILED to write artifact: " << err << std::endl;
            return 1;
        }
        std::cout << "      Artifact written successfully. Total size: " << fs::file_size(test_filepath) << " bytes." << std::endl;
    }

    // 4. Open and validate with ArtifactReader
    std::cout << "[2/5] Opening artifact and verifying framing & checksums ..." << std::endl;
    ArtifactReader reader;
    std::string err;
    if (!reader.open(test_filepath, &err)) {
        std::cerr << "FAILED to open artifact: " << err << std::endl;
        return 1;
    }

    if (!reader.validate_checksum(&err)) {
        std::cerr << "FAILED whole-file checksum validation: " << err << std::endl;
        return 1;
    }
    std::cout << "      Trailing file checksum verified successfully!" << std::endl;

    // 5. Verify Metadata round-trip
    std::cout << "[3/5] Verifying metadata round-trip ..." << std::endl;
    const auto& read_meta = reader.metadata();
    assert(read_meta.model_name == original_meta.model_name);
    assert(read_meta.quant_scheme == original_meta.quant_scheme);
    assert(read_meta.tokenizer_type == original_meta.tokenizer_type);
    assert(read_meta.properties == original_meta.properties);
    std::cout << "      Model Name: " << read_meta.model_name << std::endl;
    std::cout << "      Quant Scheme: " << read_meta.quant_scheme << std::endl;
    std::cout << "      Tokenizer: " << read_meta.tokenizer_type << std::endl;
    std::cout << "      Custom Properties Count: " << read_meta.properties.size() << std::endl;

    // Test UTF-16 surrogate pairs in metadata JSON parser
    std::string test_meta_json = R"({
        "model_name": "Test/\uD83D\uDE00-Model",
        "quant_scheme": "\uD83D\uDE80-Fast",
        "tokenizer_type": "bpe",
        "properties": {
            "description": "Hello \uD83D\uDC4B \uD83C\uDF0D"
        }
    })";
    auto opt_meta = ArtifactMetadata::from_json(test_meta_json);
    assert(opt_meta.has_value());
    assert(opt_meta->model_name == "Test/\xF0\x9F\x98\x80-Model");
    assert(opt_meta->quant_scheme == "\xF0\x9F\x9A\x80-Fast");
    assert(opt_meta->properties.at("description") == "Hello \xF0\x9F\x91\x8B \xF0\x9F\x8C\x8D");
    std::cout << "      UTF-16 surrogate pairs parsed successfully in metadata!" << std::endl;

    // Verify rejection of unpaired surrogate
    std::string bad_meta_json = "{\"model_name\": \"\\uD83D\"}";
    auto bad_opt = ArtifactMetadata::from_json(bad_meta_json);
    assert(!bad_opt.has_value());

    // 6. Verify Section Table & 64-byte alignments
    std::cout << "[4/5] Verifying section table and 64-byte alignment ..." << std::endl;
    auto section_names = reader.list_sections();
    assert(section_names.size() == 5);

    for (const auto& name : section_names) {
        auto info = reader.get_section_info(name);
        assert(info.has_value());
        assert((info->offset % SECTION_ALIGNMENT) == 0); // Must be 64-byte aligned
        std::cout << "      Section: " << name
                  << " | Offset: " << info->offset << " (aligned: " << (info->offset % 64 == 0 ? "YES" : "NO") << ")"
                  << " | Size: " << info->size << " bytes"
                  << " | Checksum: 0x" << std::hex << info->checksum << std::dec << std::endl;
    }

    // 7. Verify byte-exact equality for all sections
    std::cout << "[5/5] Verifying byte-exact payload retrieval ..." << std::endl;

    auto verify_section = [&](const std::string& name, const std::vector<uint8_t>& expected) {
        std::vector<uint8_t> read_data;
        if (!reader.read_section(name, read_data, &err)) {
            std::cerr << "FAILED to read section " << name << ": " << err << std::endl;
            return false;
        }
        if (read_data.size() != expected.size()) {
            std::cerr << "Size mismatch for " << name << ": expected " << expected.size() << ", got " << read_data.size() << std::endl;
            return false;
        }
        if (std::memcmp(read_data.data(), expected.data(), expected.size()) != 0) {
            std::cerr << "Content mismatch for section " << name << "!" << std::endl;
            return false;
        }
        return true;
    };

    assert(verify_section("model.embed_tokens.weight", embed_weights));
    assert(verify_section("model.layers.0.self_attn.q_proj.weight", q_proj_weights));
    assert(verify_section("model.layers.0.self_attn.q_proj.scales", q_proj_scales));
    assert(verify_section("tokenizer.data", tokenizer_data));
    assert(verify_section("chat_template.jinja", chat_template));

    std::cout << "      All 5 sections matched byte-for-byte with correct CRC-64 verification." << std::endl;

    // 8. Test corruption detection
    std::cout << "=== Testing corruption / truncation detection ===" << std::endl;
    reader.close();

    // Create a truncated copy
    std::string corrupt_filepath = "test_corrupt.xinfer";
    {
        std::ifstream src(test_filepath, std::ios::binary);
        std::ofstream dst(corrupt_filepath, std::ios::binary);
        std::vector<char> buf(fs::file_size(test_filepath) - 10); // truncate 10 bytes
        src.read(buf.data(), buf.size());
        dst.write(buf.data(), buf.size());
    }

    ArtifactReader corrupt_reader;
    bool corrupt_opened = corrupt_reader.open(corrupt_filepath, &err);
    if (!corrupt_opened) {
        std::cout << "      Successfully detected truncated file during open: " << err << std::endl;
    } else {
        bool valid = corrupt_reader.validate_checksum(&err);
        assert(!valid);
        std::cout << "      Successfully detected corrupted checksum: " << err << std::endl;
    }

    // Clean up temporary files
    fs::remove(test_filepath);
    fs::remove(corrupt_filepath);

    std::cout << "\n>>> ALL ROUND-TRIP TESTS PASSED SUCCESSFULLY! <<<\n" << std::endl;
    return 0;
}
