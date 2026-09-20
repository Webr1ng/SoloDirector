from pathlib import Path

from src.config import DEFAULT_CONFIG_PATH, load_config, output_paths


def test_config_loads_and_normalizes_root():
    config = load_config(DEFAULT_CONFIG_PATH)
    root = Path(config["project"]["root"])
    assert root == DEFAULT_CONFIG_PATH.parent.parent.resolve()
    assert config["events"]["stable_pose"]["duration_seconds"] == 3.0


def test_output_paths_are_project_relative(tmp_path):
    config = load_config(DEFAULT_CONFIG_PATH)
    config["project"]["root"] = str(tmp_path)
    paths = output_paths(config)
    assert paths["events_dir"] == tmp_path / "outputs/events"
    assert paths["events_dir"].is_dir()
