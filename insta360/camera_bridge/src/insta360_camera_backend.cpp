#include "camera_bridge/insta360_camera_backend.h"

namespace solo_director::camera {

namespace {
constexpr const char* kNotConfigured =
    "Official Insta360 Desktop Camera SDK is not configured. Add the competition-provided "
    "headers/libs and implement this adapter without changing the abstract CLI contract.";
}

bool Insta360CameraBackend::connect(const CameraConfig&, std::string& error) {
    status_ = {CameraState::error, "Insta360", kNotConfigured};
    error = status_.message;
    return false;
}

CameraStatus Insta360CameraBackend::status() const {
    if (status_.message.empty()) {
        return {CameraState::disconnected, "Insta360", kNotConfigured};
    }
    return status_;
}

bool Insta360CameraBackend::start_record(std::string& error) {
    error = kNotConfigured;
    return false;
}

bool Insta360CameraBackend::stop_record(std::string& error) {
    error = kNotConfigured;
    return false;
}

std::vector<CameraFile> Insta360CameraBackend::list_files(std::string& error) const {
    error = kNotConfigured;
    return {};
}

bool Insta360CameraBackend::download(const std::string&, const std::string&, std::string& error) {
    error = kNotConfigured;
    return false;
}

bool Insta360CameraBackend::latest(std::string&, std::string& error) const {
    error = kNotConfigured;
    return false;
}

}  // namespace solo_director::camera
