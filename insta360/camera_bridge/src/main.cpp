#include "camera_bridge/camera_backend.h"
#include "camera_bridge/insta360_camera_backend.h"
#include "camera_bridge/mock_camera_backend.h"

#include <iostream>
#include <memory>
#include <string>

using solo_director::camera::CameraBackend;
using solo_director::camera::CameraConfig;
using solo_director::camera::Insta360CameraBackend;
using solo_director::camera::MockCameraBackend;
using solo_director::camera::camera_state_to_string;

namespace {

void print_usage() {
    std::cout << "Usage: solo_director_camera_bridge [--backend mock|insta360] <command> [options]\n"
              << "Commands: connect, status, start_record, stop_record, list_files, latest, download\n"
              << "Options: --file <remote-name> --output <local-path>\n";
}

void print_status(const CameraBackend& backend) {
    const auto status = backend.status();
    std::cout << "{\"state\":\"" << camera_state_to_string(status.state)
              << "\",\"model\":\"" << status.model << "\",\"message\":\""
              << status.message << "\"}\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 2;
    }
    std::string backend_name = "mock";
    std::string command;
    std::string remote_file;
    std::string output_path;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--backend" && index + 1 < argc) {
            backend_name = argv[++index];
        } else if (argument == "--file" && index + 1 < argc) {
            remote_file = argv[++index];
        } else if (argument == "--output" && index + 1 < argc) {
            output_path = argv[++index];
        } else if (command.empty()) {
            command = argument;
        }
    }
    if (command.empty()) {
        print_usage();
        return 2;
    }

    std::unique_ptr<CameraBackend> backend;
    if (backend_name == "mock") {
        backend = std::make_unique<MockCameraBackend>();
    } else if (backend_name == "insta360") {
        backend = std::make_unique<Insta360CameraBackend>();
    } else {
        std::cerr << "Unknown backend: " << backend_name << "\n";
        return 2;
    }

    std::string error;
    if (command == "connect") {
        const bool ok = backend->connect(CameraConfig{}, error);
        if (!ok) {
            std::cerr << error << "\n";
            return 1;
        }
        print_status(*backend);
        return 0;
    }

    // The CLI is stateless by design. For the Mock backend we connect within each
    // invocation so smoke tests and future Python subprocess calls are predictable.
    if (backend_name == "mock" && !backend->connect(CameraConfig{}, error)) {
        std::cerr << error << "\n";
        return 1;
    }

    if (command == "status") {
        print_status(*backend);
    } else if (command == "start_record") {
        if (!backend->start_record(error)) {
            std::cerr << error << "\n";
            return 1;
        }
        print_status(*backend);
    } else if (command == "stop_record") {
        if (!backend->stop_record(error)) {
            std::cerr << error << "\n";
            return 1;
        }
        print_status(*backend);
    } else if (command == "list_files") {
        const auto files = backend->list_files(error);
        if (!error.empty()) {
            std::cerr << error << "\n";
            return 1;
        }
        for (const auto& file : files) {
            std::cout << file.name << "\t" << file.size_bytes << "\t" << file.created_at << "\n";
        }
    } else if (command == "latest") {
        if (!backend->latest(remote_file, error)) {
            std::cerr << error << "\n";
            return 1;
        }
        std::cout << remote_file << "\n";
    } else if (command == "download") {
        if (remote_file.empty() || output_path.empty()) {
            std::cerr << "download requires --file <remote-name> and --output <local-path>\n";
            return 2;
        }
        if (!backend->download(remote_file, output_path, error)) {
            std::cerr << error << "\n";
            return 1;
        }
        std::cout << output_path << "\n";
    } else {
        print_usage();
        return 2;
    }
    return 0;
}
