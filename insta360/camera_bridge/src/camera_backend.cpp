#include "camera_bridge/camera_backend.h"

namespace solo_director::camera {

std::string camera_state_to_string(CameraState state) {
    switch (state) {
        case CameraState::connected:
            return "connected";
        case CameraState::recording:
            return "recording";
        case CameraState::error:
            return "error";
        case CameraState::disconnected:
        default:
            return "disconnected";
    }
}

}  // namespace solo_director::camera
