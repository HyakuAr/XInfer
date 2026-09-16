#include "xinfer/engine.h"
#include "serve/server.h"

#include <iostream>
#include <string>
#include <csignal>
#include <atomic>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {
std::atomic<bool> g_shutdown_requested{false};

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        g_shutdown_requested.store(true);
    }
}
} // anonymous namespace

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --model <path>       Path to .xinfer artifact (default: out/qwen3_8_27b.xinfer)\n"
              << "  --host <ip>          Host address to bind to (default: 0.0.0.0)\n"
              << "  --port <port>        HTTP port to listen on (default: 8080)\n"
              << "  --model-id <name>    Model name in OpenAI responses (default: qwen3.8-27b)\n"
              << "  --chunk-size <int>   Chunk size for chunked prefill (default: 512)\n"
              << "  --max-seq-len <int>  Maximum context length for KV cache (default: 8192)\n"
              << "  --help, -h           Show this help message\n"
              << std::endl;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    std::string model_path = "out/qwen3_8_27b.xinfer";
    std::string host = "0.0.0.0";
    int port = 8080;
    std::string model_id = "qwen3.8-27b";
    int chunk_size = 512;
    int max_seq_len = 8192;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--model-id" && i + 1 < argc) {
            model_id = argv[++i];
        } else if (arg == "--chunk-size" && i + 1 < argc) {
            chunk_size = std::stoi(argv[++i]);
        } else if (arg == "--max-seq-len" && i + 1 < argc) {
            max_seq_len = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    std::cout << "========================================================\n"
              << " xinfer-serve: OpenAI-Compatible HTTP Serving Engine\n"
              << "========================================================\n"
              << " Model Artifact: " << model_path << "\n"
              << " Model ID:       " << model_id << "\n"
              << " Host:           " << host << "\n"
              << " Port:           " << port << "\n"
              << " Max Context:    " << max_seq_len << "\n"
              << "========================================================\n" << std::endl;

    xinfer::EngineConfig eng_cfg;
    eng_cfg.artifact_path = model_path;
    eng_cfg.prefer_b60 = true;
    eng_cfg.prefill_chunk_size = chunk_size;
    eng_cfg.max_seq_len = max_seq_len;

    xinfer::Engine engine;
    std::string err;
    if (!engine.load(eng_cfg, &err)) {
        std::cerr << "[xinfer-serve] Fatal error loading model: " << err << std::endl;
        return 1;
    }

    xinfer::serve::ServerConfig srv_cfg;
    srv_cfg.host = host;
    srv_cfg.port = port;
    srv_cfg.model_id = model_id;

    xinfer::serve::HttpServer server(engine);
    if (!server.start(srv_cfg, &err)) {
        std::cerr << "[xinfer-serve] Fatal error starting HTTP server: " << err << std::endl;
        return 1;
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "[xinfer-serve] Ready to accept OpenAI client requests.\n"
              << "Example:\n"
              << "  curl http://localhost:" << port << "/v1/chat/completions \\\n"
              << "    -H \"Content-Type: application/json\" \\\n"
              << "    -d '{\"model\": \"" << model_id << "\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'\n\n"
              << "Press Ctrl+C to stop server.\n" << std::endl;

    while (!g_shutdown_requested.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    std::cout << "\n[xinfer-serve] Shutdown signal received, stopping server..." << std::endl;
    server.stop();
    std::cout << "[xinfer-serve] Server stopped cleanly." << std::endl;
    return 0;
}
