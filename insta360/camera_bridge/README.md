# Insta360 Camera Bridge

This directory is the formal integration boundary between the SoloDirector AI pipeline and
an Insta360 camera. It is intentionally a standalone C++17 executable/library so the Python
side can call it through a stable CLI or, later, a small local HTTP/JSON adapter.

## Contract

The current command contract is:

```text
solo_director_camera_bridge [--backend mock|insta360] <command>

connect
status
start_record
stop_record
list_files
latest
download --file <remote-name> --output <local-path>
```

`MockCameraBackend` implements the contract without hardware and is useful for CI/smoke tests.
`Insta360CameraBackend` is a deliberate, compilable placeholder: it does not invent or imitate
official symbols. Every operation reports that the official Desktop Camera SDK is not yet
configured.

## Build and smoke test

From the repository root:

```bash
cmake -S . -B insta360/camera_bridge/build
cmake --build insta360/camera_bridge/build
insta360/camera_bridge/build/insta360/camera_bridge/solo_director_camera_bridge \
  --backend mock connect
```

Depending on the CMake generator, the executable may instead be under
`insta360/camera_bridge/build/solo_director_camera_bridge`.

## Real SDK TODO

When the competition provides the official Insta360 Desktop Camera SDK, place its headers and
libraries outside Git-tracked source (for example under a local `sdk/` directory ignored by
`.gitignore`) and implement the calls in `src/insta360_camera_backend.cpp`. Do not commit closed
source binaries until their license permits it. The adapter must preserve the abstract methods:

`connect / status / start_record / stop_record / list_files / download / latest`.

The Python API and offline video pipeline do not depend on this SDK being present. A future
bridge process can emit JSON lines or expose localhost HTTP; the current CLI is the simplest
stable seam for the competition demo.
