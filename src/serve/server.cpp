#include "server.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <mutex>

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
using socket_t = SOCKET;
#define IS_VALID_SOCKET(s) ((s) != INVALID_SOCKET)
#define CLOSE_SOCKET(s) closesocket(s)
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
using socket_t = int;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)
#define IS_VALID_SOCKET(s) ((s) >= 0)
#define CLOSE_SOCKET(s) close(s)
#endif

namespace xinfer::serve {

namespace {

std::mutex g_engine_mutex;

bool send_all(socket_t sock, const char* data, size_t len) {
    size_t total_sent = 0;
    while (total_sent < len) {
        int sent = send(sock, data + total_sent, static_cast<int>(len - total_sent), 0);
        if (sent <= 0) {
            return false;
        }
        total_sent += static_cast<size_t>(sent);
    }
    return true;
}

bool send_string(socket_t sock, const std::string& str) {
    return send_all(sock, str.data(), str.size());
}

std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

} // anonymous namespace

HttpServer::HttpServer(Engine& engine) : engine_(engine) {
#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
}

HttpServer::~HttpServer() {
    stop();
#ifdef _WIN32
    WSACleanup();
#endif
}

bool HttpServer::start(const ServerConfig& config, std::string* error_msg) {
    if (is_running_.load()) {
        if (error_msg) *error_msg = "HttpServer is already running";
        return false;
    }

    config_ = config;

    socket_t listen_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (!IS_VALID_SOCKET(listen_fd)) {
        if (error_msg) *error_msg = "Failed to create listening socket";
        return false;
    }

    int opt = 1;
#ifdef _WIN32
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt));
#else
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(static_cast<uint16_t>(config_.port));

    if (config_.host.empty() || config_.host == "0.0.0.0") {
        server_addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        if (inet_pton(AF_INET, config_.host.c_str(), &server_addr.sin_addr) <= 0) {
            CLOSE_SOCKET(listen_fd);
            if (error_msg) *error_msg = "Invalid host IP address: " + config_.host;
            return false;
        }
    }

    if (bind(listen_fd, reinterpret_cast<sockaddr*>(&server_addr), sizeof(server_addr)) == SOCKET_ERROR) {
        CLOSE_SOCKET(listen_fd);
        if (error_msg) *error_msg = "Failed to bind socket to " + config_.host + ":" + std::to_string(config_.port);
        return false;
    }

    if (listen(listen_fd, SOMAXCONN) == SOCKET_ERROR) {
        CLOSE_SOCKET(listen_fd);
        if (error_msg) *error_msg = "Failed to listen on socket";
        return false;
    }

    server_socket_ = static_cast<uintptr_t>(listen_fd);
    is_running_.store(true);

    std::cout << "[xinfer-serve] OpenAI HTTP Server listening at http://"
              << (config_.host.empty() ? "0.0.0.0" : config_.host) << ":" << config_.port
              << " (endpoints: /v1/chat/completions, /v1/models, /health)" << std::endl;

    accept_thread_ = std::thread(&HttpServer::accept_loop, this);
    return true;
}

void HttpServer::stop() {
    if (!is_running_.exchange(false)) return;

    if (server_socket_ != ~static_cast<uintptr_t>(0)) {
        socket_t s = static_cast<socket_t>(server_socket_);
        server_socket_ = ~static_cast<uintptr_t>(0);
        CLOSE_SOCKET(s);
    }

    if (accept_thread_.joinable()) {
        accept_thread_.join();
    }
}

void HttpServer::wait() {
    if (accept_thread_.joinable()) {
        accept_thread_.join();
    }
}

void HttpServer::accept_loop() {
    socket_t listen_fd = static_cast<socket_t>(server_socket_);

    while (is_running_.load()) {
        sockaddr_in client_addr{};
#ifdef _WIN32
        int addr_len = sizeof(client_addr);
#else
        socklen_t addr_len = sizeof(client_addr);
#endif
        socket_t client_fd = accept(listen_fd, reinterpret_cast<sockaddr*>(&client_addr), &addr_len);
        if (!IS_VALID_SOCKET(client_fd)) {
            if (!is_running_.load()) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // Handle client connection synchronously (serialized inference per AGENTS.md single-resident model contract)
        handle_client(static_cast<uintptr_t>(client_fd));
    }
}

void HttpServer::handle_client(uintptr_t client_socket) {
    socket_t sock = static_cast<socket_t>(client_socket);

    std::vector<char> buffer(4096);
    std::string raw_request;
    size_t header_end_pos = std::string::npos;

    // 1. Read HTTP headers until "\r\n\r\n"
    while (header_end_pos == std::string::npos) {
        int bytes = recv(sock, buffer.data(), static_cast<int>(buffer.size()), 0);
        if (bytes <= 0) {
            CLOSE_SOCKET(sock);
            return;
        }
        raw_request.append(buffer.data(), bytes);
        header_end_pos = raw_request.find("\r\n\r\n");
        if (raw_request.size() > 65536) { // Header safety limit 64KB
            std::string err = "HTTP/1.1 431 Request Header Fields Too Large\r\nConnection: close\r\n\r\n";
            send_string(sock, err);
            CLOSE_SOCKET(sock);
            return;
        }
    }

    std::string headers_str = raw_request.substr(0, header_end_pos);
    std::string body_prefix = raw_request.substr(header_end_pos + 4);

    // Parse request line
    std::istringstream h_stream(headers_str);
    std::string method, path, protocol;
    h_stream >> method >> path >> protocol;

    // Parse headers
    std::string line;
    std::getline(h_stream, line); // consume remainder of first line
    size_t content_length = 0;

    while (std::getline(h_stream, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = to_lower_copy(line.substr(0, colon));
            std::string val = line.substr(colon + 1);
            while (!val.empty() && val.front() == ' ') val.erase(val.begin());

            if (key == "content-length") {
                try {
                    content_length = std::stoull(val);
                } catch (...) {
                    content_length = 0;
                }
            }
        }
    }

    // Handle CORS preflight
    if (method == "OPTIONS") {
        std::string cors_resp =
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
            "Connection: close\r\n\r\n";
        send_string(sock, cors_resp);
        CLOSE_SOCKET(sock);
        return;
    }

    // Handle GET /health
    if (method == "GET" && (path == "/health" || path == "/v1/health")) {
        std::string body = "{\"status\":\"ok\"}";
        std::string resp =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Content-Length: " + std::to_string(body.size()) + "\r\n"
            "Connection: close\r\n\r\n" + body;
        send_string(sock, resp);
        CLOSE_SOCKET(sock);
        return;
    }

    // Handle GET /v1/models
    if (method == "GET" && (path == "/v1/models" || path == "/models")) {
        int64_t now_ts = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        std::string body = "{\"object\":\"list\",\"data\":[{\"id\":\"" + config_.model_id +
                           "\",\"object\":\"model\",\"created\":" + std::to_string(now_ts) +
                           ",\"owned_by\":\"xinfer\"}]}";
        std::string resp =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Content-Length: " + std::to_string(body.size()) + "\r\n"
            "Connection: close\r\n\r\n" + body;
        send_string(sock, resp);
        CLOSE_SOCKET(sock);
        return;
    }

    // Route only POST /v1/chat/completions (or /chat/completions)
    if (method != "POST" || (path != "/v1/chat/completions" && path != "/chat/completions")) {
        ApiError err;
        err.status_code = 404;
        err.type = "invalid_request_error";
        err.code = "not_found";
        err.message = "The requested endpoint '" + path + "' does not exist. Use POST /v1/chat/completions.";
        std::string err_body = err.to_json();
        std::string resp =
            "HTTP/1.1 404 Not Found\r\n"
            "Content-Type: application/json\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Content-Length: " + std::to_string(err_body.size()) + "\r\n"
            "Connection: close\r\n\r\n" + err_body;
        send_string(sock, resp);
        CLOSE_SOCKET(sock);
        return;
    }

    // 2. Read remainder of POST body
    if (content_length > config_.max_request_size) {
        ApiError err;
        err.status_code = 413;
        err.type = "invalid_request_error";
        err.code = "request_entity_too_large";
        err.message = "Request body exceeds max configured size.";
        std::string err_body = err.to_json();
        std::string resp = "HTTP/1.1 413 Payload Too Large\r\nContent-Type: application/json\r\nContent-Length: " +
                           std::to_string(err_body.size()) + "\r\nConnection: close\r\n\r\n" + err_body;
        send_string(sock, resp);
        CLOSE_SOCKET(sock);
        return;
    }

    std::string body = std::move(body_prefix);
    while (body.size() < content_length) {
        size_t needed = content_length - body.size();
        size_t chunk_bytes = std::min(needed, buffer.size());
        int bytes = recv(sock, buffer.data(), static_cast<int>(chunk_bytes), 0);
        if (bytes <= 0) break;
        body.append(buffer.data(), bytes);
    }

    // 3. Parse OpenAI chat completion request
    ChatCompletionRequest req;
    ApiError parse_err;
    if (!parse_chat_completion_request(body, req, parse_err)) {
        std::string err_body = parse_err.to_json();
        std::string resp =
            "HTTP/1.1 " + std::to_string(parse_err.status_code) + " Bad Request\r\n"
            "Content-Type: application/json\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Content-Length: " + std::to_string(err_body.size()) + "\r\n"
            "Connection: close\r\n\r\n" + err_body;
        send_string(sock, resp);
        CLOSE_SOCKET(sock);
        return;
    }

    std::string prompt = req.format_prompt();
    GenerationConfig gen_cfg;
    gen_cfg.max_new_tokens = req.max_tokens;
    gen_cfg.temperature = req.temperature;
    gen_cfg.apply_chat_template = false; // already formatted via format_prompt()

    // Validate prompt tokens and max_tokens against model context length
    size_t prompt_tokens = engine_.count_tokens(prompt, false);
    std::string context_err;
    if (!engine_.validate_tokens(prompt_tokens, req.max_tokens, &context_err)) {
        ApiError err = make_context_length_exceeded_error(engine_.max_seq_len(), prompt_tokens, req.max_tokens);
        std::string err_body = err.to_json();
        std::string resp =
            "HTTP/1.1 400 Bad Request\r\n"
            "Content-Type: application/json\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Content-Length: " + std::to_string(err_body.size()) + "\r\n"
            "Connection: close\r\n\r\n" + err_body;
        send_string(sock, resp);
        CLOSE_SOCKET(sock);
        return;
    }

    std::string req_id = generate_completion_id();
    int64_t created_ts = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // 4. Execute inference serialized via public Engine interface
    std::lock_guard<std::mutex> lock(g_engine_mutex);

    if (!req.stream) {
        // --- Non-Streaming Response ---
        auto result = engine_.generate(prompt, gen_cfg);
        if (!result.success) {
            ApiError err;
            err.status_code = 400;
            err.type = "invalid_request_error";
            err.code = result.error_code.empty() ? "bad_request" : result.error_code;
            err.message = result.error_msg;
            std::string err_body = err.to_json();
            std::string http_resp =
                "HTTP/1.1 400 Bad Request\r\n"
                "Content-Type: application/json\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Content-Length: " + std::to_string(err_body.size()) + "\r\n"
                "Connection: close\r\n\r\n" + err_body;
            send_string(sock, http_resp);
            CLOSE_SOCKET(sock);
            return;
        }

        ChatCompletionResponse resp;
        resp.id = req_id;
        resp.created = created_ts;
        resp.model = config_.model_id;

        ChatChoice choice;
        choice.index = 0;
        choice.message.role = "assistant";
        choice.message.content = result.text;
        choice.finish_reason = result.finish_reason;
        resp.choices.push_back(std::move(choice));

        resp.usage.prompt_tokens = result.prompt_tokens;
        resp.usage.completion_tokens = result.generated_tokens;
        resp.usage.total_tokens = result.prompt_tokens + result.generated_tokens;

        std::string resp_json = resp.to_json();
        std::string http_resp =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Content-Length: " + std::to_string(resp_json.size()) + "\r\n"
            "Connection: close\r\n\r\n" + resp_json;
        send_string(sock, http_resp);

    } else {
        // --- Streaming SSE Response ---
        std::string sse_headers =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: close\r\n"
            "Access-Control-Allow-Origin: *\r\n\r\n";
        send_string(sock, sse_headers);

        // Initial chunk announcing role
        ChatCompletionChunk init_chunk;
        init_chunk.id = req_id;
        init_chunk.created = created_ts;
        init_chunk.model = config_.model_id;
        ChunkChoice init_choice;
        init_choice.index = 0;
        init_choice.delta.role = "assistant";
        init_choice.delta.content = "";
        init_chunk.choices.push_back(std::move(init_choice));
        send_string(sock, init_chunk.to_sse_event());

        // Token callback streaming delta pieces
        auto token_callback = [&](const std::string& piece, int64_t /*tok_id*/) -> bool {
            ChatCompletionChunk chunk;
            chunk.id = req_id;
            chunk.created = created_ts;
            chunk.model = config_.model_id;
            ChunkChoice ch;
            ch.index = 0;
            ch.delta.content = piece;
            chunk.choices.push_back(std::move(ch));
            return send_string(sock, chunk.to_sse_event());
        };

        auto result = engine_.generate(prompt, gen_cfg, token_callback);

        // Final finish_reason chunk
        ChatCompletionChunk finish_chunk;
        finish_chunk.id = req_id;
        finish_chunk.created = created_ts;
        finish_chunk.model = config_.model_id;
        ChunkChoice fin_choice;
        fin_choice.index = 0;
        fin_choice.finish_reason = result.finish_reason;
        finish_chunk.choices.push_back(std::move(fin_choice));
        send_string(sock, finish_chunk.to_sse_event());

        // End of stream indicator
        send_string(sock, "data: [DONE]\n\n");
    }

    CLOSE_SOCKET(sock);
}

} // namespace xinfer::serve
