#include "camera_bridge/mock_camera_backend.h"

#include <fstream>

namespace solo_director::camera {

bool MockCameraBackend::connect(const CameraConfig&, std::string&) {
    state_ = CameraState::connected;
    files_ = {{"mock_latest.mp4", 1024, "2026-01-01T00:00:00Z"}};
    return true;
}

CameraStatus MockCameraBackend::status() const {
    return {state_, "MockInsta360", "Mock backend; no physical camera is connected"};
}

bool MockCameraBackend::start_record(std::string& error) {
    if (state_ != CameraState::connected) {
        error = "connect before start_record";
        return false;
    }
    state_ = CameraState::recording;
    return true;
}

bool MockCameraBackend::stop_record(std::string& error) {
    if (state_ != CameraState::recording) {
        error = "camera is not recording";
        return false;
    }
    state_ = CameraState::connected;
    return true;
}

std::vector<CameraFile> MockCameraBackend::list_files(std::string&) const { return files_; }

bool MockCameraBackend::download(const std::string& remote_name, const std::string& local_path,
                                 std::string& error) {
    for (const auto& file : files_) {
        if (file.name == remote_name) {
            std::ofstream output(local_path, std::ios::binary);
            if (!output) {
                error = "could not open local output: " + local_path;
                return false;
            }
            output << "Mock camera payload for " << remote_name << "\n";
            return true;
        }
    }
    error = "remote file not found: " + remote_name;
    return false;
}

bool MockCameraBackend::latest(std::string& remote_name, std::string& error) const {
    if (files_.empty()) {
        error = "no mock files available";
        return false;
    }
    remote_name = files_.back().name;
    return true;
}

}  // namespace solo_director::camera
