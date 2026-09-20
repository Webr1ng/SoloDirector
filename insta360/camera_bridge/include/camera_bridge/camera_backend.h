#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace solo_director::camera {

enum class CameraState { disconnected, connected, recording, error };

struct CameraConfig {
    std::string connection = "usb";
    int timeout_ms = 5000;
};

struct CameraStatus {
    CameraState state = CameraState::disconnected;
    std::string model;
    std::string message;
};

struct CameraFile {
    std::string name;
    std::uint64_t size_bytes = 0;
    std::string created_at;
};

class CameraBackend {
public:
    virtual ~CameraBackend() = default;

    virtual bool connect(const CameraConfig& config, std::string& error) = 0;
    virtual CameraStatus status() const = 0;
    virtual bool start_record(std::string& error) = 0;
    virtual bool stop_record(std::string& error) = 0;
    virtual std::vector<CameraFile> list_files(std::string& error) const = 0;
    virtual bool download(const std::string& remote_name, const std::string& local_path,
                          std::string& error) = 0;
    virtual bool latest(std::string& remote_name, std::string& error) const = 0;
};

std::string camera_state_to_string(CameraState state);

}  // namespace solo_director::camera
