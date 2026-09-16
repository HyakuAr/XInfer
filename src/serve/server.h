#pragma once

#include "protocol.h"
#include "xinfer/engine.h"

#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <cstdint>

namespace xinfer::serve {

struct ServerConfig {
    std::string host{"0.0.0.0"};
    int         port{8080};
    size_t      max_request_size{10 * 1024 * 1024}; // 10MB
    std::string model_id{"qwen3.8-27b"};
};

class HttpServer {
public:
    explicit HttpServer(Engine& engine);
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    // Start server on configured host and port (non-blocking, launches worker loop)
    bool start(const ServerConfig& config, std::string* error_msg = nullptr);

    // Stop listening and close active connections
    void stop();

    // Query if server is active and listening
    bool is_running() const noexcept { return is_running_.load(); }

    // Block caller until server stops (e.g. on SIGINT)
    void wait();

private:
    void accept_loop();
    void handle_client(uintptr_t client_socket);

    Engine& engine_;
    ServerConfig config_;
    std::atomic<bool> is_running_{false};
    uintptr_t server_socket_{~static_cast<uintptr_t>(0)}; // INVALID_SOCKET
    std::thread accept_thread_;
};

} // namespace xinfer::serve
