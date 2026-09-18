#pragma once

#include "protocol.h"
#include "xinfer/engine.h"

#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <cstdint>

namespace xinfer::serve {

struct ServerConfig {
    std::string host{"0.0.0.0"};
    int         port{8080};
    size_t      max_request_size{10 * 1024 * 1024}; // 10MB
    std::string model_id{"qwen3.8-27b"};
    size_t      num_workers{8};                     // Concurrency contract: 1-8 held connections (inference serialized via single resident model)
    size_t      max_queued_requests{32};            // Bounded request backlog before returning 503
    int         recv_timeout_sec{15};               // Socket recv timeout in seconds (default 15s)
};

class HttpServer {
public:
    explicit HttpServer(Engine& engine);
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    // Start server on configured host and port (non-blocking, launches worker pool and accept loop)
    bool start(const ServerConfig& config, std::string* error_msg = nullptr);

    // Stop listening and close active connections
    void stop();

    // Query if server is active and listening
    bool is_running() const noexcept { return is_running_.load(); }

    // Query actual listening port (useful if port 0 was passed for ephemeral assignment)
    int port() const noexcept { return config_.port; }

    // Block caller until server stops (e.g. on SIGINT)
    void wait();

private:
    void accept_loop();
    void worker_loop();
    void handle_client(uintptr_t client_socket);

    Engine& engine_;
    ServerConfig config_;
    std::atomic<bool> is_running_{false};
    uintptr_t server_socket_{~static_cast<uintptr_t>(0)}; // INVALID_SOCKET
    std::thread accept_thread_;

    std::vector<std::thread> worker_threads_;
    std::queue<uintptr_t> client_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    std::mutex engine_mutex_;
};

} // namespace xinfer::serve
