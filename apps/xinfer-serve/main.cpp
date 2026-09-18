#include "xinfer/engine.h"
#include "serve/server.h"
#include <sycl/sycl.hpp>
#include <iostream>
#include <string>
#include <csignal>
#include <atomic>
#include <new>
#include <exception>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
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
              << "  --model <path>              Path to .xinfer artifact (default: out/qwen3_8_27b.xinfer)\n"
              << "  --tokenizer-path <path>     Path to external tokenizer.json (if not embedded in artifact)\n"
              << "  --chat-template-path <path> Path to external chat_template.jinja (if not embedded in artifact)\n"
              << "  --host <ip>                 Host address to bind to (default: 0.0.0.0)\n"
              << "  --port <port>               HTTP port to listen on (default: 8080)\n"
              << "  --model-id <name>           Model name in OpenAI responses (default: qwen3.8-27b)\n"
              << "  --chunk-size <int>          Chunk size for chunked prefill (default: 512)\n"
              << "  --max-seq-len <int>         Maximum context length for KV cache (default: 8192)\n"
              << "  --workers <int>             Number of worker threads (1-8, default: 8)\n"
              << "  --recv-timeout <int>        Socket receive timeout in seconds (default: 15)\n"
              << "  --skip-checksum             Skip whole-file CRC-64 checksum validation\n"
              << "  --help, -h                  Show this help message\n"
              << std::endl;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    std::string model_path = "out/qwen3_8_27b.xinfer";
    std::string tokenizer_path;
    std::string chat_template_path;
    std::string host = "0.0.0.0";
    int port = 8080;
    std::string model_id = "qwen3.8-27b";
    int chunk_size = 512;
    int max_seq_len = 8192;
    int workers = 8;
    int recv_timeout = 15;
    bool validate_checksum = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--tokenizer-path" && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (arg == "--chat-template-path" && i + 1 < argc) {
            chat_template_path = argv[++i];
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
        } else if (arg == "--workers" && i + 1 < argc) {
            workers = std::clamp(std::stoi(argv[++i]), 1, 8);
        } else if (arg == "--recv-timeout" && i + 1 < argc) {
            recv_timeout = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--skip-checksum" || arg == "--no-checksum") {
            validate_checksum = false;
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
              << " Checksum:       " << (validate_checksum ? "verify (CRC-64)" : "skipped") << "\n";
    if (!tokenizer_path.empty()) {
        std::cout << " Tokenizer:      " << tokenizer_path << "\n";
    }
    std::cout << "========================================================\n" << std::endl;

    xinfer::EngineConfig eng_cfg;
    eng_cfg.artifact_path = model_path;
    eng_cfg.tokenizer_path = tokenizer_path;
    eng_cfg.chat_template_path = chat_template_path;
    eng_cfg.prefer_b60 = true;
    eng_cfg.prefill_chunk_size = chunk_size;
    eng_cfg.max_seq_len = max_seq_len;
    eng_cfg.validate_checksum = validate_checksum;

    xinfer::Engine engine;
    std::string err;
    try {
        if (!engine.load(eng_cfg, &err)) {
            std::cerr << "[xinfer-serve] Fatal error loading model: " << err << std::endl;
            return 1;
        }
    } catch (const sycl::exception& e) {
        std::cerr << "[xinfer-serve] Fatal SYCL exception loading model: " << e.what() << std::endl;
        return 1;
    } catch (const std::bad_alloc& e) {
        std::cerr << "[xinfer-serve] Fatal out-of-memory (std::bad_alloc) loading model: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[xinfer-serve] Fatal exception loading model: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "[xinfer-serve] Fatal unknown exception loading model" << std::endl;
        return 1;
    }

    xinfer::serve::ServerConfig srv_cfg;
    srv_cfg.host = host;
    srv_cfg.port = port;
    srv_cfg.model_id = model_id;
    srv_cfg.num_workers = static_cast<size_t>(workers);
    srv_cfg.recv_timeout_sec = recv_timeout;

    xinfer::serve::HttpServer server(engine);
    try {
        if (!server.start(srv_cfg, &err)) {
            std::cerr << "[xinfer-serve] Fatal error starting HTTP server: " << err << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "[xinfer-serve] Fatal exception starting HTTP server: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "[xinfer-serve] Fatal unknown exception starting HTTP server" << std::endl;
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
