from dataclasses import dataclass, field


@dataclass
class InputConfig:
    projection: str = ""
    analysis_width: int = 1536


@dataclass
class ProjectionConfig:
    num_views: int = 4
    view_fov_deg: float = 100.0
    view_size: int = 640


@dataclass
class DetectConfig:
    model: str = "yolov8n.pt"
    conf: float = 0.35
    classes: tuple[int, ...] = (0,)
    detect_fps: float = 5.0
    device: str | None = None


@dataclass
class TrackConfig:
    iou_threshold: float = 0.3
    max_age: int = 30
    min_track_len: int = 1


@dataclass
class IdentityConfig:
    enabled: bool = False
    n_people: int = 0
    distance_threshold: float = 0.35
    samples_per_track: int = 5


@dataclass
class RegionConfig:
    window_sec: float = 8.0
    stride_sec: float = 4.0
    gap_deg: float = 35.0
    flat_gap: float = 0.15
    context_deg: float = 12.0
    min_fov_deg: float = 70.0
    max_fov_deg: float = 125.0
    size: int = 960
    fps: int = 10


@dataclass
class PanelConfig:
    sample_fps: float = 2.0
    frame_limit: int = 32
    frame_width: int = 960
    max_candidates: int = 100
    max_api_calls: int = 100
    workers: int = 5
    timeout_sec: int = 120
    consensus_threshold: float = 0.5
    reliability_file: str | None = None
    allow_cloud_upload: bool = False
    selected_expert_ids: tuple[str, ...] | None = None


@dataclass
class ExportConfig:
    out_dir: str = "output"
    export_video: bool = True
    live_photo_sec: float = 3.0
    make_reel: bool = True
    size: int = 1440
    fps: int = 30


@dataclass
class Config:
    input_path: str = ""
    prepare_only: bool = False
    input: InputConfig = field(default_factory=InputConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    region: RegionConfig = field(default_factory=RegionConfig)
    panel: PanelConfig = field(default_factory=PanelConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
