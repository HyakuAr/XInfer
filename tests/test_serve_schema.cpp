#include "serve/protocol.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace xinfer::serve;

void test_parse_valid_single_turn_request() {
    std::cout << "[Test 1/5] Parse valid single-turn request..." << std::endl;
    std::string json = R"({
        "model": "qwen3.8-27b",
        "messages": [
            {"role": "user", "content": "Hello world!"}
        ],
        "max_tokens": 64,
        "temperature": 0.7,
        "stream": false
    })";

    ChatCompletionRequest req;
    ApiError err;
    bool ok = parse_chat_completion_request(json, req, err);
    if (!ok) {
        std::cerr << "Failed to parse valid request: " << err.message << std::endl;
        std::exit(1);
    }

    assert(req.model == "qwen3.8-27b");
    assert(req.messages.size() == 1);
    assert(req.messages[0].role == "user");
    assert(req.messages[0].content == "Hello world!");
    assert(req.max_tokens == 64);
    assert(std::abs(req.temperature - 0.7f) < 1e-4);
    assert(!req.stream);

    std::string prompt = req.format_prompt();
    std::string expected_prompt = "<|im_start|>user\nHello world!<|im_end|>\n<|im_start|>assistant\n";
    assert(prompt == expected_prompt);
    std::cout << "  -> PASSED: Single-turn request parsed and formatted." << std::endl;
}

void test_parse_multi_turn_with_system() {
    std::cout << "[Test 2/5] Parse multi-turn request with system prompt and streaming..." << std::endl;
    std::string json = R"({
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4."},
            {"role": "user", "content": "Multiply by 3."}
        ],
        "max_tokens": 128,
        "stream": true
    })";

    ChatCompletionRequest req;
    ApiError err;
    bool ok = parse_chat_completion_request(json, req, err);
    if (!ok) {
        std::cerr << "Failed to parse multi-turn request: " << err.message << std::endl;
        std::exit(1);
    }

    assert(req.messages.size() == 4);
    assert(req.messages[0].role == "system");
    assert(req.messages[0].content == "Be concise.");
    assert(req.messages[1].role == "user");
    assert(req.messages[2].role == "assistant");
    assert(req.messages[3].role == "user");
    assert(req.max_tokens == 128);
    assert(req.stream);

    std::string prompt = req.format_prompt();
    assert(prompt.find("<|im_start|>system\nBe concise.<|im_end|>\n") != std::string::npos);
    assert(prompt.find("<|im_start|>assistant\n4.<|im_end|>\n") != std::string::npos);
    assert(prompt.rfind("<|im_start|>assistant\n") == prompt.size() - 22);
    std::cout << "  -> PASSED: Multi-turn prompt properly formatted." << std::endl;
}

void test_parse_invalid_requests() {
    std::cout << "[Test 3/5] Parse invalid requests (error handling)..." << std::endl;

    ChatCompletionRequest req;
    ApiError err;

    // 1. Empty body
    assert(!parse_chat_completion_request("", req, err));
    assert(err.status_code == 400);

    // 2. Not an object
    assert(!parse_chat_completion_request("[\"not an object\"]", req, err));
    assert(err.status_code == 400);

    // 3. Missing messages
    assert(!parse_chat_completion_request("{\"model\": \"qwen3.8-27b\"}", req, err));
    assert(err.code == "missing_required_field");

    // 4. Empty messages array
    assert(!parse_chat_completion_request("{\"messages\": []}", req, err));
    assert(err.code == "missing_required_field");

    // 5. Message missing role
    assert(!parse_chat_completion_request("{\"messages\": [{\"content\": \"test\"}]}", req, err));
    assert(err.code == "missing_role");

    // Verify error JSON serialization
    std::string err_json = err.to_json();
    assert(err_json.find("\"error\":{") != std::string::npos);
    assert(err_json.find("\"code\":\"missing_role\"") != std::string::npos);

    std::cout << "  -> PASSED: All invalid request schemas rejected with HTTP 400 and OpenAI error format." << std::endl;
}

void test_response_json_serialization() {
    std::cout << "[Test 4/5] Non-streaming ChatCompletionResponse JSON serialization..." << std::endl;

    ChatCompletionResponse resp;
    resp.id = "chatcmpl-test12345678";
    resp.created = 1726480000;
    resp.model = "qwen3.8-27b";

    ChatChoice choice;
    choice.index = 0;
    choice.message.role = "assistant";
    choice.message.content = "Rayleigh scattering causes the sky to appear blue.";
    choice.finish_reason = "stop";
    resp.choices.push_back(choice);

    resp.usage.prompt_tokens = 24;
    resp.usage.completion_tokens = 10;
    resp.usage.total_tokens = 34;

    std::string json = resp.to_json();

    // Verify JSON structure by re-parsing
    JsonValue root;
    std::string parse_err;
    bool ok = parse_json(json, root, parse_err);
    if (!ok) {
        std::cerr << "Failed to re-parse generated response JSON: " << parse_err << "\nJSON: " << json << std::endl;
        std::exit(1);
    }

    assert(root.get_string("id") == "chatcmpl-test12345678");
    assert(root.get_string("object") == "chat.completion");
    assert(root.get_int("created") == 1726480000);
    assert(root.get_string("model") == "qwen3.8-27b");

    const auto* choices = root.find("choices");
    assert(choices && choices->type == JsonValue::Type::Array && choices->arr_val.size() == 1);
    const auto& c0 = choices->arr_val[0];
    assert(c0.get_int("index") == 0);
    assert(c0.get_string("finish_reason") == "stop");

    const auto* msg = c0.find("message");
    assert(msg && msg->get_string("role") == "assistant");
    assert(msg->get_string("content") == "Rayleigh scattering causes the sky to appear blue.");

    const auto* usage = root.find("usage");
    assert(usage && usage->get_int("prompt_tokens") == 24);
    assert(usage && usage->get_int("completion_tokens") == 10);
    assert(usage && usage->get_int("total_tokens") == 34);

    std::cout << "  -> PASSED: ChatCompletionResponse matches OpenAI schema." << std::endl;
}

void test_streaming_sse_event_serialization() {
    std::cout << "[Test 5/5] Streaming SSE chunk serialization..." << std::endl;

    ChatCompletionChunk chunk;
    chunk.id = "chatcmpl-stream123";
    chunk.created = 1726480001;
    chunk.model = "qwen3.8-27b";

    ChunkChoice ch;
    ch.index = 0;
    ch.delta.content = " quantum";
    chunk.choices.push_back(ch);

    std::string sse = chunk.to_sse_event();
    assert(sse.rfind("data: ", 0) == 0);
    assert(sse.size() >= 4 && sse.substr(sse.size() - 2) == "\n\n");

    std::string json_part = sse.substr(6, sse.size() - 8);
    JsonValue root;
    std::string err;
    assert(parse_json(json_part, root, err));
    assert(root.get_string("object") == "chat.completion.chunk");
    assert(root.get_string("id") == "chatcmpl-stream123");

    const auto* choices = root.find("choices");
    assert(choices && choices->arr_val.size() == 1);
    const auto* delta = choices->arr_val[0].find("delta");
    assert(delta && delta->get_string("content") == " quantum");

    std::cout << "  -> PASSED: Streaming SSE chunk conforms to OpenAI streaming specification." << std::endl;
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << " xinfer Milestone 9 OpenAI Serving Protocol & Schema Test" << std::endl;
    std::cout << "==========================================================" << std::endl;

    test_parse_valid_single_turn_request();
    test_parse_multi_turn_with_system();
    test_parse_invalid_requests();
    test_response_json_serialization();
    test_streaming_sse_event_serialization();

    std::cout << "==========================================================" << std::endl;
    std::cout << " ALL MILESTONE 9 SCHEMA TESTS PASSED!" << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}
