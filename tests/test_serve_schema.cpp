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

    std::cout << "  -> PASSED: Single-turn request parsed." << std::endl;
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

    std::cout << "  -> PASSED: Multi-turn request parsed." << std::endl;
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
    bool parse_ok = parse_json(json_part, root, err);
    assert(parse_ok);
    if (!parse_ok) {
        std::cerr << "Failed to parse SSE JSON: " << err << "\nJSON: " << json_part << std::endl;
        std::exit(1);
    }
    assert(root.get_string("object") == "chat.completion.chunk");
    assert(root.get_string("id") == "chatcmpl-stream123");

    const auto* choices = root.find("choices");
    assert(choices && choices->arr_val.size() == 1);
    const auto* delta = choices->arr_val[0].find("delta");
    assert(delta && delta->get_string("content") == " quantum");

    std::cout << "  -> PASSED: Streaming SSE chunk conforms to OpenAI streaming specification." << std::endl;
}

void test_context_length_exceeded_error() {
    std::cout << "[Test 6/6] Context length exceeded error serialization..." << std::endl;

    // 1. Prompt itself exceeds max_seq_len
    ApiError err1 = make_context_length_exceeded_error(8192, 9000, 128);
    assert(err1.status_code == 400);
    assert(err1.type == "invalid_request_error");
    assert(err1.code == "context_length_exceeded");
    assert(err1.param == "messages");
    std::string json1 = err1.to_json();
    assert(json1.find("\"code\":\"context_length_exceeded\"") != std::string::npos);
    assert(json1.find("\"param\":\"messages\"") != std::string::npos);
    assert(json1.find("9000 tokens") != std::string::npos);

    // 2. Prompt + max_tokens exceeds max_seq_len
    ApiError err2 = make_context_length_exceeded_error(8192, 8000, 500);
    assert(err2.status_code == 400);
    assert(err2.code == "context_length_exceeded");
    std::string json2 = err2.to_json();
    assert(json2.find("8500 tokens") != std::string::npos);
    assert(json2.find("8000 in the messages, 500 in the completion") != std::string::npos);

    std::cout << "  -> PASSED: context_length_exceeded matches OpenAI error schema." << std::endl;
}

void test_utf16_surrogate_pairs() {
    std::cout << "[Test 7/8] Parse UTF-16 surrogate pairs and unicode escapes..." << std::endl;

    // 1. Valid emoji / astral plane: \uD83D\uDE00 -> 😀 (U+1F600, UTF-8: \xF0\x9F\x98\x80)
    std::string json1 = "{\"emoji\": \"\\uD83D\\uDE00\"}";
    JsonValue v1;
    std::string err1;
    assert(parse_json(json1, v1, err1));
    assert(v1.get_string("emoji") == "\xF0\x9F\x98\x80");

    // 2. Musical symbol G clef: \uD834\uDD1E -> 𝄞 (U+1D11E, UTF-8: \xF0\x9D\x84\x9E)
    std::string json2 = "{\"clef\": \"\\uD834\\uDD1E\"}";
    JsonValue v2;
    std::string err2;
    assert(parse_json(json2, v2, err2));
    assert(v2.get_string("clef") == "\xF0\x9D\x84\x9E");

    // 3. Mixed standard unicode and surrogate pairs:
    // \u0041 (A) + \u00E9 (é) + \u4E2D (中) + \uD83D\uDE80 (🚀)
    std::string json3 = "{\"text\": \"\\u0041\\u00E9\\u4E2D\\uD83D\\uDE80\"}";
    JsonValue v3;
    std::string err3;
    assert(parse_json(json3, v3, err3));
    assert(v3.get_string("text") == "A\xC3\xA9\xE4\xB8\xAD\xF0\x9F\x9A\x80");

    // 4. In request body messages:
    std::string req_json = "{\"messages\": [{\"role\": \"user\", \"content\": \"Hello \\uD83D\\uDC4B\\uD83C\\uDF0D\"}]}";
    ChatCompletionRequest req;
    ApiError api_err;
    bool parse_req_ok = parse_chat_completion_request(req_json, req, api_err);
    assert(parse_req_ok);
    assert(req.messages.size() == 1);
    // 👋 (U+1F44B: \xF0\x9F\x91\x8B) + 🌍 (U+1F30D: \xF0\x9F\x8C\x8D)
    assert(req.messages[0].content == "Hello \xF0\x9F\x91\x8B\xF0\x9F\x8C\x8D");

    // 5. Error case: unpaired high surrogate
    std::string bad_json1 = "{\"str\": \"\\uD83D\"}";
    JsonValue bad_v1;
    std::string bad_err1;
    assert(!parse_json(bad_json1, bad_v1, bad_err1));

    // 6. Error case: unpaired low surrogate
    std::string bad_json2 = "{\"str\": \"\\uDE00\"}";
    JsonValue bad_v2;
    std::string bad_err2;
    assert(!parse_json(bad_json2, bad_v2, bad_err2));

    std::cout << "  -> PASSED: UTF-16 surrogate pairs and astral characters properly decoded to UTF-8." << std::endl;
}

void test_top_p_parsing() {
    std::cout << "[Test 8/8] Parse top_p parameter in ChatCompletionRequest..." << std::endl;

    // Default top_p is 1.0
    std::string json_def = "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}";
    ChatCompletionRequest req_def;
    ApiError err_def;
    bool parse_def_ok = parse_chat_completion_request(json_def, req_def, err_def);
    assert(parse_def_ok);
    assert(std::abs(req_def.top_p - 1.0f) < 1e-4);

    // Explicit top_p
    std::string json_p = "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"top_p\": 0.95}";
    ChatCompletionRequest req_p;
    ApiError err_p;
    bool parse_p_ok = parse_chat_completion_request(json_p, req_p, err_p);
    assert(parse_p_ok);
    assert(std::abs(req_p.top_p - 0.95f) < 1e-4);

    std::cout << "  -> PASSED: top_p parameter correctly parsed." << std::endl;
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
    test_context_length_exceeded_error();
    test_utf16_surrogate_pairs();
    test_top_p_parsing();

    std::cout << "==========================================================" << std::endl;
    std::cout << " ALL MILESTONE 9 SCHEMA TESTS PASSED!" << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}
