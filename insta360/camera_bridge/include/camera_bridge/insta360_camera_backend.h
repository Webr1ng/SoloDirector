#pragma once

#include "camera_backend.h"

namespace solo_director::camera {

// This adapter intentionally contains no guessed Insta360 symbols. The official
// Desktop Camera SDK headers/libs supplied by the competition must be wired here.
class Insta360CameraBackend final : public CameraBackend {
public:
    bool connect(const CameraConfig& config, std::string& error) override;
    CameraStatus status() const override;
    bool start_record(std::string& error) override;
    bool stop_record(std::string& error) override;
    std::vector<CameraFile> list_files(std::string& error) const override;
    bool download(const std::string& remote_name, const std::string& local_path,
                  std::string& error) override;
    bool latest(std::string& remote_name, std::string& error) const override;

private:
    CameraStatus status_{};
};

}  // namespace solo_director::camera
