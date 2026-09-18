#include "serve/protocol.h"
#include "serve/server.h"
#include "xinfer/engine.h"
#include "core/device.h"
#include "targets/qwen3_8/chat_template.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <thread>
#include <chrono>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
using test_socket_t = SOCKET;
#define TEST_CLOSE_SOCKET(s) closesocket(s)
#define TEST_IS_VALID(s) ((s) != INVALID_SOCKET)
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
using test_socket_t = int;
#define TEST_CLOSE_SOCKET(s) close(s)
#define TEST_IS_VALID(s) ((s) >= 0)
#endif

using namespace xinfer;
using namespace xinfer::serve;
using namespace xinfer::core;
using namespace xinfer::targets::qwen3_8;

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

    // 6. max_tokens <= 0 or invalid
    assert(!parse_chat_completion_request("{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"max_tokens\": 0}", req, err));
    assert(err.status_code == 400);
    assert(err.code == "invalid_parameter");
    assert(err.param == "max_tokens");

    assert(!parse_chat_completion_request("{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"max_tokens\": -5}", req, err));
    assert(err.status_code == 400);
    assert(err.code == "invalid_parameter");
    assert(err.param == "max_tokens");

    assert(!parse_chat_completion_request("{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"max_tokens\": \"not_a_number\"}", req, err));
    assert(err.status_code == 400);
    assert(err.code == "invalid_parameter");
    assert(err.param == "max_tokens");

    // Verify error JSON serialization
    std::string err_json = err.to_json();
    assert(err_json.find("\"error\":{") != std::string::npos);
    assert(err_json.find("\"code\":\"invalid_parameter\"") != std::string::npos);
    assert(err_json.find("\"param\":\"max_tokens\"") != std::string::npos);

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

void test_reasoning_content_serialization() {
    std::cout << "[Test 9/9] Reasoning content serialization in response and streaming SSE..." << std::endl;

    // 1. Non-streaming ChatCompletionResponse with reasoning_content
    ChatCompletionResponse resp;
    resp.id = "chatcmpl-reasoning123";
    resp.created = 1726480000;
    resp.model = "qwen3.8-27b";

    ChatChoice choice;
    choice.index = 0;
    choice.message.role = "assistant";
    choice.message.content = "42 is the answer.";
    choice.message.reasoning_content = "Thinking about the meaning of life...";
    choice.finish_reason = "stop";
    resp.choices.push_back(choice);

    std::string json = resp.to_json();
    JsonValue root;
    std::string err;
    assert(parse_json(json, root, err));
    const auto* choices = root.find("choices");
    assert(choices && choices->arr_val.size() == 1);
    const auto* msg = choices->arr_val[0].find("message");
    assert(msg && msg->get_string("content") == "42 is the answer.");
    assert(msg && msg->get_string("reasoning_content") == "Thinking about the meaning of life...");

    // 2. Streaming ChatCompletionChunk with delta.reasoning_content
    ChatCompletionChunk chunk;
    chunk.id = "chatcmpl-stream-reasoning";
    chunk.created = 1726480001;
    chunk.model = "qwen3.8-27b";

    ChunkChoice ch;
    ch.index = 0;
    ch.delta.reasoning_content = " step 1";
    chunk.choices.push_back(ch);

    std::string sse = chunk.to_sse_event();
    assert(sse.rfind("data: ", 0) == 0);
    std::string json_part = sse.substr(6, sse.size() - 8);
    JsonValue sse_root;
    assert(parse_json(json_part, sse_root, err));
    const auto* sse_choices = sse_root.find("choices");
    assert(sse_choices && sse_choices->arr_val.size() == 1);
    const auto* delta = sse_choices->arr_val[0].find("delta");
    assert(delta && delta->get_string("reasoning_content") == " step 1");

    std::cout << "  -> PASSED: reasoning_content correctly serialized in non-streaming and streaming schemas." << std::endl;
}

test_socket_t connect_to_server(int port) {
    test_socket_t s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (!TEST_IS_VALID(s)) return s;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        TEST_CLOSE_SOCKET(s);
        return static_cast<test_socket_t>(INVALID_SOCKET);
    }
    return s;
}

std::string send_and_recv_response(test_socket_t s, const std::string& req) {
    send(s, req.data(), static_cast<int>(req.size()), 0);
    std::string resp;
    char buf[1024];
    while (true) {
        int r = recv(s, buf, sizeof(buf), 0);
        if (r <= 0) break;
        resp.append(buf, r);
    }
    TEST_CLOSE_SOCKET(s);
    return resp;
}

void test_server_request_hardening() {
    std::cout << "[Test 10/10] Server request hardening (chunked transfer rejection, max_tokens <= 0, recv timeout)..." << std::endl;

    // 1. Validate Engine::validate_tokens with max_new_tokens <= 0
    xinfer::Engine engine;
    std::string v_err;
    assert(!engine.validate_tokens(10, 0, &v_err));
    assert(v_err.find("Invalid max_new_tokens") != std::string::npos);
    assert(!engine.validate_tokens(10, -1, &v_err));
    assert(v_err.find("Invalid max_new_tokens") != std::string::npos);

    // 2. Validate Engine::generate with max_new_tokens <= 0
    xinfer::GenerationConfig bad_cfg;
    bad_cfg.max_new_tokens = 0;
    auto bad_gen = engine.generate("hello", bad_cfg);
    assert(!bad_gen.success);

    // 3. Start HttpServer on ephemeral port with 2-second timeout
    ServerConfig cfg;
    cfg.host = "127.0.0.1";
    cfg.port = 0; // OS assigns free ephemeral port
    cfg.recv_timeout_sec = 2;
    HttpServer server(engine);
    std::string start_err;
    bool started = server.start(cfg, &start_err);
    assert(started);
    int port = server.port();
    assert(port > 0);

    // 4. Test Transfer-Encoding: chunked rejection
    {
        test_socket_t s = connect_to_server(port);
        assert(TEST_IS_VALID(s));
        std::string req =
            "POST /v1/chat/completions HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Transfer-Encoding: chunked\r\n"
            "Content-Type: application/json\r\n\r\n"
            "1e\r\n"
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}\r\n"
            "0\r\n\r\n";
        std::string resp = send_and_recv_response(s, req);
        assert(resp.find("HTTP/1.1 400 Bad Request") != std::string::npos);
        assert(resp.find("\"code\":\"unsupported_parameter\"") != std::string::npos);
        assert(resp.find("\"param\":\"Transfer-Encoding\"") != std::string::npos);
        assert(resp.find("Chunked transfer encoding is not supported") != std::string::npos);
    }

    // 5. Test max_tokens: 0 rejection by server
    {
        test_socket_t s = connect_to_server(port);
        assert(TEST_IS_VALID(s));
        std::string body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":0}";
        std::string req =
            "POST /v1/chat/completions HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
        std::string resp = send_and_recv_response(s, req);
        assert(resp.find("HTTP/1.1 400 Bad Request") != std::string::npos);
        assert(resp.find("\"code\":\"invalid_parameter\"") != std::string::npos);
        assert(resp.find("\"param\":\"max_tokens\"") != std::string::npos);
    }

    // 6. Test partial body with timeout handling
    {
        test_socket_t s = connect_to_server(port);
        assert(TEST_IS_VALID(s));
        std::string req =
            "POST /v1/chat/completions HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: 200\r\n\r\n"
            "{\"messages\":[";
        send(s, req.data(), static_cast<int>(req.size()), 0);
        std::string resp;
        char buf[1024];
        while (true) {
            int r = recv(s, buf, sizeof(buf), 0);
            if (r <= 0) break;
            resp.append(buf, r);
        }
        TEST_CLOSE_SOCKET(s);
        assert(resp.find("408 Request Timeout") != std::string::npos || resp.empty());
    }

    server.stop();
    std::cout << "  -> PASSED: Server request hardening verified (chunked rejection, max_tokens <= 0 rejection, and recv timeout)." << std::endl;
}

void test_base64_chunked_decoder() {
    std::cout << "[Test 11/16] Chunked Base64 Decoder..." << std::endl;
    // 1. Valid base64: "Hello, World!" -> "SGVsbG8sIFdvcmxkIQ=="
    std::string b64 = "SGVsbG8sIFdvcmxkIQ==";
    std::vector<uint8_t> bytes;
    std::string err;
    assert(decode_base64_chunked(b64, bytes, &err));
    std::string decoded(bytes.begin(), bytes.end());
    assert(decoded == "Hello, World!");

    // 2. Data URL prefix
    std::string data_url = "data:image/png;base64,SGVsbG8sIFdvcmxkIQ==";
    bytes.clear();
    assert(decode_base64_chunked(data_url, bytes, &err));
    decoded.assign(bytes.begin(), bytes.end());
    assert(decoded == "Hello, World!");

    // 3. Invalid base64 character
    bytes.clear();
    assert(!decode_base64_chunked("SGVs!@#$%", bytes, &err));
    assert(!err.empty());

    std::cout << "  -> PASSED: Chunked Base64 Decoder verified." << std::endl;
}

void test_multipart_parser() {
    std::cout << "[Test 12/16] Multipart/Form-Data Parser..." << std::endl;
    std::string boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW";
    std::string body =
        "------WebKitFormBoundary7MA4YWxkTrZu0gW\r\n"
        "Content-Disposition: form-data; name=\"messages\"\r\n\r\n"
        "{\"messages\": [{\"role\": \"user\", \"content\": \"What is in this image?\"}]}\r\n"
        "------WebKitFormBoundary7MA4YWxkTrZu0gW\r\n"
        "Content-Disposition: form-data; name=\"image\"; filename=\"test.png\"\r\n"
        "Content-Type: image/png\r\n\r\n"
        "FAKEDATABYTES\r\n"
        "------WebKitFormBoundary7MA4YWxkTrZu0gW--\r\n";

    ChatCompletionRequest req;
    ApiError err;
    assert(parse_multipart_chat_completion_request(body, boundary, req, err));
    assert(req.messages.size() == 1);
    assert(req.messages[0].role == "user");
    // Notice automatic <image> tag prepend since user didn't explicitly include <image>
    assert(req.messages[0].content.find("<image>") != std::string::npos);
    assert(req.images.size() == 1);
    assert(req.images[0].format == "image/png");
    std::string img_data(req.images[0].raw_bytes.begin(), req.images[0].raw_bytes.end());
    assert(img_data == "FAKEDATABYTES");

    std::cout << "  -> PASSED: Multipart/Form-Data Parser verified." << std::endl;
}

void test_chat_template_image_expansion() {
    std::cout << "[Test 13/16] Chat Template <image> Expansion to 256 <|image_pad|> Tokens..." << std::endl;
    targets::qwen3_8::QwenChatTemplate tmpl;
    std::vector<ChatMessage> msgs = {
        ChatMessage{.role = "user", .content = "Describe this picture: <image>", .reasoning_content = ""}
    };
    targets::qwen3_8::ChatTemplateOptions opts;
    opts.patches_per_image = 256;
    opts.image_pad_token = "<|image_pad|>";
    std::string rendered = tmpl.render(msgs, opts);

    // Count occurrences of "<|image_pad|>"
    size_t count = 0;
    size_t pos = 0;
    while ((pos = rendered.find("<|image_pad|>", pos)) != std::string::npos) {
        count++;
        pos += std::string_view("<|image_pad|>").size();
    }
    assert(count == 256);
    assert(rendered.find("<image>") == std::string::npos);

    std::cout << "  -> PASSED: Chat Template expanded <image> to exactly 256 <|image_pad|> tokens." << std::endl;
}

void test_image_dimension_validation() {
    std::cout << "[Test 14/16] Image Dimension & Fail-Loud Contract Validation..." << std::endl;
    // Valid dimensions
    std::string err;
    assert(core::validate_image_dimensions(224, 224, 1024, 4.0f, &err));
    assert(core::validate_image_dimensions(1024, 1024, 1024, 4.0f, &err));
    assert(core::validate_image_dimensions(800, 200, 1024, 4.0f, &err)); // 4:1 aspect ratio
    assert(core::validate_image_dimensions(200, 800, 1024, 4.0f, &err)); // 1:4 aspect ratio

    // Exceed max resolution (> 1024)
    assert(!core::validate_image_dimensions(1025, 512, 1024, 4.0f, &err));
    assert(err.find("exceeds maximum allowed resolution") != std::string::npos);

    assert(!core::validate_image_dimensions(512, 1200, 1024, 4.0f, &err));
    assert(err.find("exceeds maximum allowed resolution") != std::string::npos);

    // Exceed max aspect ratio (> 4:1)
    assert(!core::validate_image_dimensions(900, 150, 1024, 4.0f, &err)); // 6:1 aspect ratio
    assert(err.find("exceeds maximum allowed aspect ratio") != std::string::npos);

    assert(!core::validate_image_dimensions(150, 900, 1024, 4.0f, &err)); // 1:6 aspect ratio
    assert(err.find("exceeds maximum allowed aspect ratio") != std::string::npos);

    // PNG Header parsing
    uint8_t mock_png[30] = {
        0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A, // signature
        0x00, 0x00, 0x00, 0x0D, // length
        'I', 'H', 'D', 'R',     // type
        0x00, 0x00, 0x01, 0x00, // width = 256 (0x100)
        0x00, 0x00, 0x00, 0x80, // height = 128 (0x80)
        0x08, 0x02, 0x00, 0x00, 0x00
    };
    core::ImageDimensions dims;
    assert(core::parse_image_dimensions(mock_png, sizeof(mock_png), dims, &err));
    assert(dims.width == 256);
    assert(dims.height == 128);

    // BMP Header parsing
    uint8_t mock_bmp[30] = {
        'B', 'M', // signature
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // file header
        0, 0, 0, 0, // DIB header size
        0x40, 0x01, 0x00, 0x00, // width = 320 (0x140)
        0xF0, 0x00, 0x00, 0x00  // height = 240 (0xF0)
    };
    assert(core::parse_image_dimensions(mock_bmp, sizeof(mock_bmp), dims, &err));
    assert(dims.width == 320);
    assert(dims.height == 240);

    std::cout << "  -> PASSED: Image dimension parsing & fail-loud bounds verified." << std::endl;
}

void test_server_fail_loud_image_resolution() {
    std::cout << "[Test 15/16] Server HTTP 400 Fail-Loud for Oversized Image / Aspect Ratio..." << std::endl;
    Engine mock_engine;
    HttpServer server(mock_engine);
    ServerConfig cfg;
    cfg.port = 0;
    cfg.num_workers = 1;
    cfg.recv_timeout_sec = 2;
    std::string err_msg;
    assert(server.start(cfg, &err_msg));
    int port = server.port();
    assert(port > 0);

    // 1. Send multipart request with oversized PNG (2000x2000)
    {
        uint8_t oversized_png[30] = {
            0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D,
            'I', 'H', 'D', 'R',
            0x00, 0x00, 0x07, 0xD0, // width = 2000
            0x00, 0x00, 0x07, 0xD0, // height = 2000
            0x08, 0x02, 0x00, 0x00, 0x00
        };
        std::string img_bytes(reinterpret_cast<char*>(oversized_png), sizeof(oversized_png));
        std::string boundary = "----TestBoundary123";
        std::string body =
            "------TestBoundary123\r\n"
            "Content-Disposition: form-data; name=\"messages\"\r\n\r\n"
            "{\"messages\": [{\"role\": \"user\", \"content\": \"Inspect this\"}]}\r\n"
            "------TestBoundary123\r\n"
            "Content-Disposition: form-data; name=\"image\"; filename=\"huge.png\"\r\n"
            "Content-Type: image/png\r\n\r\n" +
            img_bytes + "\r\n"
            "------TestBoundary123--\r\n";

        std::string req =
            "POST /v1/chat/completions HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Content-Type: multipart/form-data; boundary=----TestBoundary123\r\n"
            "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;

        test_socket_t s = connect_to_server(port);
        assert(TEST_IS_VALID(s));
        std::string resp = send_and_recv_response(s, req);
        assert(resp.find("400 Bad Request") != std::string::npos);
        assert(resp.find("image_resolution_exceeded") != std::string::npos ||
               resp.find("exceeds maximum allowed resolution") != std::string::npos);
    }

    // 2. Send multipart request with invalid aspect ratio (1000x100 = 10:1)
    {
        uint8_t wide_png[30] = {
            0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D,
            'I', 'H', 'D', 'R',
            0x00, 0x00, 0x03, 0xE8, // width = 1000
            0x00, 0x00, 0x00, 0x64, // height = 100 (10:1 ratio)
            0x08, 0x02, 0x00, 0x00, 0x00
        };
        std::string img_bytes(reinterpret_cast<char*>(wide_png), sizeof(wide_png));
        std::string boundary = "----TestBoundary123";
        std::string body =
            "------TestBoundary123\r\n"
            "Content-Disposition: form-data; name=\"messages\"\r\n\r\n"
            "{\"messages\": [{\"role\": \"user\", \"content\": \"Inspect this\"}]}\r\n"
            "------TestBoundary123\r\n"
            "Content-Disposition: form-data; name=\"image\"; filename=\"wide.png\"\r\n"
            "Content-Type: image/png\r\n\r\n" +
            img_bytes + "\r\n"
            "------TestBoundary123--\r\n";

        std::string req =
            "POST /v1/chat/completions HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Content-Type: multipart/form-data; boundary=----TestBoundary123\r\n"
            "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;

        test_socket_t s = connect_to_server(port);
        assert(TEST_IS_VALID(s));
        std::string resp = send_and_recv_response(s, req);
        assert(resp.find("400 Bad Request") != std::string::npos);
        assert(resp.find("image_aspect_ratio_exceeded") != std::string::npos ||
               resp.find("exceeds maximum allowed aspect ratio") != std::string::npos);
    }

    server.stop();
    std::cout << "  -> PASSED: Server HTTP 400 Bad Request fail-loud contract verified." << std::endl;
}

void test_usm_image_preprocessing() {
    std::cout << "[Test 16/16] GPU USM Image Preprocessing (Bilinear Resize & Normalization)..." << std::endl;
    auto ctx = core::DeviceContext::create(true);
    int64_t src_w = 64;
    int64_t src_h = 64;
    int64_t channels = 3;
    std::vector<uint8_t> pixels(src_w * src_h * channels);
    for (size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = static_cast<uint8_t>(i % 256);
    }

    sycl::half* d_preprocessed = core::preprocess_image_to_usm(*ctx, pixels.data(), src_w, src_h, channels, 224, 224, false);
    assert(d_preprocessed != nullptr);

    size_t total_elements = 256 * 588;
    std::vector<sycl::half> h_out(total_elements);
    ctx->copy_device_to_host(h_out.data(), d_preprocessed, total_elements * sizeof(sycl::half), true);

    for (size_t i = 0; i < 100; ++i) {
        float v = static_cast<float>(h_out[i]);
        assert(v >= -1.05f && v <= 1.05f);
    }

    ctx->free_device(d_preprocessed);
    std::cout << "  -> PASSED: GPU USM Image Preprocessing verified." << std::endl;
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
    test_reasoning_content_serialization();
    test_server_request_hardening();
    test_base64_chunked_decoder();
    test_multipart_parser();
    test_chat_template_image_expansion();
    test_image_dimension_validation();
    test_server_fail_loud_image_resolution();
    test_usm_image_preprocessing();

    std::cout << "==========================================================" << std::endl;
    std::cout << " ALL MILESTONE 9 SCHEMA & MULTIMODAL TESTS PASSED!" << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}


