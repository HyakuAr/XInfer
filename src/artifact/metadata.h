#pragma once

#include <string>
#include <map>
#include <string_view>
#include <optional>

namespace xinfer::artifact {

struct ArtifactMetadata {
    std::string model_name;
    std::string quant_scheme;
    std::string tokenizer_type;
    std::map<std::string, std::string> properties;

    std::string to_json() const;
    static std::optional<ArtifactMetadata> from_json(std::string_view json_str);

    bool operator==(const ArtifactMetadata& other) const {
        return model_name == other.model_name &&
               quant_scheme == other.quant_scheme &&
               tokenizer_type == other.tokenizer_type &&
               properties == other.properties;
    }
};

} // namespace xinfer::artifact
