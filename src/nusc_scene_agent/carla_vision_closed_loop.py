from __future__ import annotations

import csv
import json
import math
import os
import queue
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from nusc_scene_agent.bench2drive_e2e import (
    DEFAULT_BENCH2DRIVE_CAMERAS,
    DEFAULT_BENCH2DRIVE_OUTPUT,
    VisionE2EModelConfig,
    _build_vision_e2e_model,
    _normalize_images_on_device,
    _require_torch_stack,
)
from nusc_scene_agent.carla_closed_loop import (
    DEFAULT_CARLA_ROOT,
    PurePursuitConfig,
    _add_carla_python_paths,
    build_carla_launch_command,
    format_carla_launch_command,
    project_to_route,
    pure_pursuit_control,
)
from nusc_scene_agent.geometry import normalize_angle


DEFAULT_CARLA_VISION_OUTPUT = Path("outputs/carla_vision_closed_loop_v1")
CARLA_VISION_CLOSED_LOOP_SCHEMA = "carla_vision_closed_loop_v1"
CARLA_VISION_BATCH_REPORT_NAME = "carla_vision_batch_report.json"
CARLA_VISION_BATCH_MARKDOWN_NAME = "carla_vision_batch_report.md"

BENCH2DRIVE_DT_S = 0.5
LANE_ERROR_FAILURE_M = 4.0
MODEL_WAYPOINT_LOOKAHEAD_M = 7.0
MODEL_STRAIGHT_PATH_RIGHT_M = 0.12
MODEL_STEER_DEADBAND = 0.025
MODEL_STEER_DAMPING_BAND = 0.070
MIN_DEFAULT_ROUTE_LENGTH_M = 40.0
MAX_DEFAULT_ROUTE_LENGTH_M = 220.0
PREFERRED_DEFAULT_ROUTE_LENGTH_M = 0.0
CAMERA_TRANSFORMS = {
    "rgb_front": {"x": 1.45, "y": 0.0, "z": 1.65, "pitch": 0.0, "yaw": 0.0, "roll": 0.0},
    "rgb_front_left": {"x": 1.35, "y": -0.45, "z": 1.65, "pitch": 0.0, "yaw": -55.0, "roll": 0.0},
    "rgb_front_right": {"x": 1.35, "y": 0.45, "z": 1.65, "pitch": 0.0, "yaw": 55.0, "roll": 0.0},
    "rgb_back": {"x": -1.35, "y": 0.0, "z": 1.65, "pitch": 0.0, "yaw": 180.0, "roll": 0.0},
    "rgb_back_left": {"x": -1.15, "y": -0.45, "z": 1.65, "pitch": 0.0, "yaw": -135.0, "roll": 0.0},
    "rgb_back_right": {"x": -1.15, "y": 0.45, "z": 1.65, "pitch": 0.0, "yaw": 135.0, "roll": 0.0},
}
DEFAULT_CARLA_VISION_SCENARIOS = [
    {
        "name": "free_drive",
        "scenario_type": "free_drive",
        "spawn_index": 0,
        "destination_index": -1,
        "target_speed_mps": 5.2,
    },
    {
        "name": "pedestrian_crossing",
        "scenario_type": "pedestrian_crossing",
        "spawn_index": 0,
        "destination_index": -1,
        "target_speed_mps": 4.8,
        "progress_m": 24.0,
        "lateral_offset_m": -5.5,
        "actor_speed_mps": 1.1,
    },
    {
        "name": "dense_follow_overtake",
        "scenario_type": "dense_follow_overtake",
        "spawn_index": 0,
        "destination_index": -1,
        "target_speed_mps": 5.0,
        "ambient_vehicle_count": 28,
    },
    {
        "name": "adjacent_lane_cut_in",
        "scenario_type": "adjacent_lane_cut_in",
        "spawn_index": 0,
        "destination_index": -1,
        "target_speed_mps": 5.0,
        "ambient_vehicle_count": 24,
    },
]


@dataclass(frozen=True)
class CarlaVisionClosedLoopConfig:
    carla_root: Path = DEFAULT_CARLA_ROOT
    checkpoint_path: Path = DEFAULT_BENCH2DRIVE_OUTPUT / "vision_e2e_planner_best.pt"
    output_dir: Path = DEFAULT_CARLA_VISION_OUTPUT
    host: str = "127.0.0.1"
    port: int = 2000
    town: str = ""
    spawn_index: int = 0
    destination_index: int = -1
    route_sampling_resolution_m: float = 2.0
    route_min_length_m: float = MIN_DEFAULT_ROUTE_LENGTH_M
    route_max_length_m: float = MAX_DEFAULT_ROUTE_LENGTH_M
    route_preferred_length_m: float = PREFERRED_DEFAULT_ROUTE_LENGTH_M
    fps: int = 10
    horizon_s: float = 30.0
    warmup_ticks: int = 8
    image_size: int = 160
    camera_width: int = 320
    camera_height: int = 180
    video_width: int = 0
    video_height: int = 0
    camera_fov: float = 90.0
    carla_quality_level: str = "Epic"
    scenario_type: str = "free_drive"
    scenario_name: str = ""
    scenario_params: Optional[Mapping[str, Any]] = None
    scenarios: Optional[Sequence[Mapping[str, Any]]] = None
    target_speed_mps: float = 7.0
    brake_probability_threshold: float = 0.75
    enable_scenario_safety_override: bool = True
    enable_lane_departure_guard: bool = True
    condition_ego_route_traffic_lights: bool = False
    device: str = ""
    auto_launch: bool = False
    cuda_visible_devices: str = ""
    traffic_manager_port: int = 8000
    seed: int = 7
    graphics_adapter: int = -1
    launch_timeout_s: float = 90.0
    rpc_timeout_s: float = 30.0
    keep_server: bool = False
    video_fps: int = 10
    video_encoder: str = "hevc_nvenc"
    video_nvenc_preset: str = "p4"
    video_quality: int = 23
    render_gif: bool = True


@dataclass(frozen=True)
class CarlaVisionScenarioSpec:
    name: str
    scenario_type: str
    spawn_index: int = 0
    destination_index: int = -1
    target_speed_mps: Optional[float] = None
    parameters: Optional[Mapping[str, Any]] = None


@dataclass
class CarlaScenarioRuntime:
    name: str
    scenario_type: str
    actors: List[Any]
    metadata: Dict[str, Any]


def run_carla_vision_closed_loop(
    *,
    carla_root: Path = DEFAULT_CARLA_ROOT,
    checkpoint_path: Path = DEFAULT_BENCH2DRIVE_OUTPUT / "vision_e2e_planner_best.pt",
    output_dir: Path = DEFAULT_CARLA_VISION_OUTPUT,
    host: str = "127.0.0.1",
    port: int = 2000,
    town: str = "",
    spawn_index: int = 0,
    destination_index: int = -1,
    route_sampling_resolution_m: float = 2.0,
    route_min_length_m: float = MIN_DEFAULT_ROUTE_LENGTH_M,
    route_max_length_m: float = MAX_DEFAULT_ROUTE_LENGTH_M,
    route_preferred_length_m: float = PREFERRED_DEFAULT_ROUTE_LENGTH_M,
    fps: int = 10,
    horizon_s: float = 30.0,
    warmup_ticks: int = 8,
    image_size: int = 160,
    camera_width: int = 320,
    camera_height: int = 180,
    video_width: int = 0,
    video_height: int = 0,
    camera_fov: float = 90.0,
    carla_quality_level: str = "Epic",
    scenario_type: str = "free_drive",
    scenario_name: str = "",
    scenario_params: Optional[Mapping[str, Any]] = None,
    scenarios: Optional[Sequence[Mapping[str, Any]]] = None,
    target_speed_mps: float = 7.0,
    brake_probability_threshold: float = 0.75,
    enable_scenario_safety_override: bool = True,
    enable_lane_departure_guard: bool = True,
    condition_ego_route_traffic_lights: bool = False,
    device: str = "",
    auto_launch: bool = False,
    cuda_visible_devices: str = "",
    traffic_manager_port: int = 8000,
    seed: int = 7,
    graphics_adapter: int = -1,
    launch_timeout_s: float = 90.0,
    rpc_timeout_s: float = 30.0,
    keep_server: bool = False,
    video_fps: int = 10,
    video_encoder: str = "hevc_nvenc",
    video_nvenc_preset: str = "p4",
    video_quality: int = 23,
    render_gif: bool = True,
) -> Dict[str, Any]:
    config = CarlaVisionClosedLoopConfig(
        carla_root=Path(carla_root),
        checkpoint_path=Path(checkpoint_path),
        output_dir=Path(output_dir),
        host=str(host),
        port=int(port),
        town=str(town),
        spawn_index=int(spawn_index),
        destination_index=int(destination_index),
        route_sampling_resolution_m=float(route_sampling_resolution_m),
        route_min_length_m=float(route_min_length_m),
        route_max_length_m=float(route_max_length_m),
        route_preferred_length_m=float(route_preferred_length_m),
        fps=int(fps),
        horizon_s=float(horizon_s),
        warmup_ticks=int(warmup_ticks),
        image_size=int(image_size),
        camera_width=int(camera_width),
        camera_height=int(camera_height),
        video_width=int(video_width),
        video_height=int(video_height),
        camera_fov=float(camera_fov),
        carla_quality_level=str(carla_quality_level),
        scenario_type=str(scenario_type),
        scenario_name=str(scenario_name),
        scenario_params=dict(scenario_params or {}),
        scenarios=list(scenarios) if scenarios is not None else None,
        target_speed_mps=float(target_speed_mps),
        brake_probability_threshold=float(brake_probability_threshold),
        enable_scenario_safety_override=bool(enable_scenario_safety_override),
        enable_lane_departure_guard=bool(enable_lane_departure_guard),
        condition_ego_route_traffic_lights=bool(condition_ego_route_traffic_lights),
        device=str(device),
        auto_launch=bool(auto_launch),
        cuda_visible_devices=str(cuda_visible_devices),
        traffic_manager_port=int(traffic_manager_port),
        seed=int(seed),
        graphics_adapter=int(graphics_adapter),
        launch_timeout_s=float(launch_timeout_s),
        rpc_timeout_s=float(rpc_timeout_s),
        keep_server=bool(keep_server),
        video_fps=int(video_fps),
        video_encoder=str(video_encoder),
        video_nvenc_preset=str(video_nvenc_preset),
        video_quality=int(video_quality),
        render_gif=bool(render_gif),
    )
    return _run_carla_vision_closed_loop_config(config)


def _run_carla_vision_closed_loop_config(config: CarlaVisionClosedLoopConfig) -> Dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_specs = _normalize_scenario_specs(config.scenarios) if config.scenarios else []
    if scenario_specs and config.auto_launch and not config.keep_server:
        return _run_carla_vision_batch_with_fresh_servers(config, scenario_specs)

    torch, _, _ = _require_torch_stack(require_pillow=True)
    target_device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, model_config = _load_bench2drive_planner(torch, config.checkpoint_path, target_device)

    _add_carla_python_paths(Path(config.carla_root))
    import carla  # type: ignore

    server_process: Optional[subprocess.Popen[Any]] = None
    if config.auto_launch and not _can_connect_to_carla(carla, config.host, config.port, timeout_s=3.0):
        try:
            server_process = _launch_carla_server(config)
            _wait_for_carla(
                carla,
                config.host,
                config.port,
                timeout_s=config.launch_timeout_s,
                process=server_process,
            )
        except Exception:
            if server_process is not None:
                _terminate_carla_process(server_process)
            raise

    try:
        client = carla.Client(config.host, int(config.port))
        startup_timeout_s = min(max(float(config.rpc_timeout_s), 5.0), 45.0)
        client.set_timeout(startup_timeout_s)
        world = _resolve_carla_world(client, config.town)
        client.set_timeout(max(float(config.rpc_timeout_s), 5.0))
        traffic_manager = _configure_traffic_manager(client, config)
        map_name = str(world.get_map().name).rsplit("/", 1)[-1]
        original_settings = world.get_settings()
    except Exception:
        if server_process is not None:
            _terminate_carla_process(server_process)
        raise
    try:
        _configure_carla_world(world, config)
        if scenario_specs:
            batch_started_at = time.time()
            reports = []
            for spec in scenario_specs:
                scenario_output = output_dir / _safe_filename(spec.name)
                scenario_config = CarlaVisionClosedLoopConfig(
                    **{
                        **dict(config.__dict__),
                        "output_dir": scenario_output,
                        "spawn_index": int(spec.spawn_index),
                        "destination_index": int(spec.destination_index),
                        "scenario_type": str(spec.scenario_type),
                        "scenario_name": str(spec.name),
                        "scenario_params": dict(spec.parameters or {}),
                        "target_speed_mps": float(spec.target_speed_mps)
                        if spec.target_speed_mps is not None
                        else float(config.target_speed_mps),
                        "scenarios": None,
                    }
                )
                reports.append(
                    _run_carla_vision_scenario_in_world(
                        carla=carla,
                        client=client,
                        world=world,
                        map_name=map_name,
                        model=model,
                        torch=torch,
                        target_device=target_device,
                        model_config=model_config,
                        traffic_manager=traffic_manager,
                        config=scenario_config,
                    )
                )
            summary = _build_carla_vision_batch_summary(
                output_dir=output_dir,
                carla_root=config.carla_root,
                checkpoint_path=config.checkpoint_path,
                map_name=map_name,
                requested_town=config.town,
                reports=reports,
                started_at=batch_started_at,
            )
            (output_dir / CARLA_VISION_BATCH_REPORT_NAME).write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )
            _write_carla_vision_batch_markdown(summary, output_dir / CARLA_VISION_BATCH_MARKDOWN_NAME)
            return summary
        return _run_carla_vision_scenario_in_world(
            carla=carla,
            client=client,
            world=world,
            map_name=map_name,
            model=model,
            torch=torch,
            target_device=target_device,
            model_config=model_config,
            traffic_manager=traffic_manager,
            config=config,
        )
    finally:
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass
        if server_process is not None and not config.keep_server:
            _terminate_carla_process(server_process)


def _run_carla_vision_batch_with_fresh_servers(
    config: CarlaVisionClosedLoopConfig,
    scenario_specs: Sequence[CarlaVisionScenarioSpec],
) -> Dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_started_at = time.time()
    reports = []
    for scenario_idx, spec in enumerate(scenario_specs):
        scenario_output = output_dir / _safe_filename(spec.name)
        scenario_kwargs = {
            **dict(config.__dict__),
            "output_dir": scenario_output,
            "port": int(config.port) + int(scenario_idx),
            "traffic_manager_port": int(config.traffic_manager_port) + int(scenario_idx),
            "spawn_index": int(spec.spawn_index),
            "destination_index": int(spec.destination_index),
            "scenario_type": str(spec.scenario_type),
            "scenario_name": str(spec.name),
            "scenario_params": dict(spec.parameters or {}),
            "target_speed_mps": float(spec.target_speed_mps)
            if spec.target_speed_mps is not None
            else float(config.target_speed_mps),
            "scenarios": None,
        }
        reports.append(run_carla_vision_closed_loop(**scenario_kwargs))
        time.sleep(2.0)
    map_name = str(reports[0].get("town") or config.town or "") if reports else str(config.town or "")
    summary = _build_carla_vision_batch_summary(
        output_dir=output_dir,
        carla_root=config.carla_root,
        checkpoint_path=config.checkpoint_path,
        map_name=map_name,
        requested_town=config.town,
        reports=reports,
        started_at=batch_started_at,
    )
    (output_dir / CARLA_VISION_BATCH_REPORT_NAME).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _write_carla_vision_batch_markdown(summary, output_dir / CARLA_VISION_BATCH_MARKDOWN_NAME)
    return summary


def _run_carla_vision_scenario_in_world(
    *,
    carla: Any,
    client: Any,
    world: Any,
    map_name: str,
    model: Any,
    torch: Any,
    target_device: Any,
    model_config: VisionE2EModelConfig,
    traffic_manager: Optional[Any],
    config: CarlaVisionClosedLoopConfig,
) -> Dict[str, Any]:
    started_at = time.time()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    actors: List[Any] = []
    sensors: List[Any] = []
    camera_queues: Dict[str, "queue.Queue[Any]"] = {}
    video_queue: Optional["queue.Queue[Any]"] = None
    collision_events: List[Dict[str, Any]] = []
    scenario_runtime = CarlaScenarioRuntime(
        name=config.scenario_name or config.scenario_type,
        scenario_type=config.scenario_type,
        actors=[],
        metadata={},
    )

    try:
        carla_map = world.get_map()
        route_constraints = _route_constraints_from_params(dict(config.scenario_params or {}))
        route_trace, route_points, spawn_transform, _ = _select_carla_route(
            carla=carla,
            carla_map=carla_map,
            spawn_index=config.spawn_index,
            destination_index=config.destination_index,
            sampling_resolution_m=config.route_sampling_resolution_m,
            min_route_length_m=config.route_min_length_m,
            max_route_length_m=config.route_max_length_m,
            preferred_route_length_m=config.route_preferred_length_m,
            **route_constraints,
        )
        ego = _spawn_ego_vehicle(carla=carla, world=world, spawn_transform=spawn_transform)
        actors.append(ego)
        scenario_runtime = _spawn_scenario_runtime(
            carla=carla,
            client=client,
            world=world,
            carla_map=carla_map,
            traffic_manager=traffic_manager,
            route_points=route_points,
            config=config,
        )
        _condition_ego_route_traffic_lights(
            carla=carla,
            world=world,
            ego=ego,
            route_points=route_points,
            runtime=scenario_runtime,
            enabled=bool(config.condition_ego_route_traffic_lights),
        )
        actors.extend(scenario_runtime.actors)
        sensors, camera_queues = _attach_rgb_cameras(
            carla=carla,
            world=world,
            ego=ego,
            config=config,
        )
        video_sensor, video_queue = _attach_video_camera(
            carla=carla,
            world=world,
            ego=ego,
            config=config,
        )
        if video_sensor is not None:
            sensors.append(video_sensor)
        collision_sensor = _attach_collision_sensor(carla=carla, world=world, ego=ego, events=collision_events)
        sensors.append(collision_sensor)
        actors.extend(sensors)

        for step_idx in range(max(int(config.warmup_ticks), 0)):
            _update_scenario_runtime(
                carla=carla,
                runtime=scenario_runtime,
                step_idx=step_idx,
                traffic_manager=traffic_manager,
                carla_map=carla_map,
                ego=ego,
                route_points=route_points,
            )
            _condition_ego_route_traffic_lights(
                carla=carla,
                world=world,
                ego=ego,
                route_points=route_points,
                runtime=scenario_runtime,
                enabled=bool(config.condition_ego_route_traffic_lights),
            )
            world.tick()
            _drain_camera_queues(camera_queues)
            if video_queue is not None:
                _drain_queue(video_queue)

        states, frames = _rollout_vision_planner(
            carla=carla,
            world=world,
            ego=ego,
            model=model,
            torch=torch,
            device=target_device,
            model_config=model_config,
            route_trace=route_trace,
            route_points=route_points,
            camera_queues=camera_queues,
            video_queue=video_queue,
            scenario_runtime=scenario_runtime,
            traffic_manager=traffic_manager,
            carla_map=carla_map,
            collision_events=collision_events,
            config=config,
        )
    finally:
        for actor in reversed(actors):
            try:
                if hasattr(actor, "stop"):
                    actor.stop()
                actor.destroy()
            except Exception:
                pass

    route_length = _route_length(route_points)
    metrics = _evaluate_carla_vision_rollout(
        states=states,
        route_points=route_points,
        collision_events=collision_events,
    )
    control_attribution = _summarize_carla_vision_control_attribution(states, scenario_runtime)
    media = _write_carla_vision_outputs(
        output_dir=output_dir,
        states=states,
        frames=frames,
        route_points=route_points,
        config=config,
        metrics=metrics,
    )
    report = {
        "schema": CARLA_VISION_CLOSED_LOOP_SCHEMA,
        "output_dir": str(output_dir),
        "carla_root": str(config.carla_root),
        "checkpoint_path": str(config.checkpoint_path),
        "host": config.host,
        "port": int(config.port),
        "town": map_name,
        "requested_town": config.town,
        "scenario": {
            "name": scenario_runtime.name,
            "type": scenario_runtime.scenario_type,
            "actor_count": len(scenario_runtime.actors),
            "metadata": scenario_runtime.metadata,
        },
        "spawn_index": int(config.spawn_index),
        "destination_index": int(config.destination_index),
        "route_length_m": route_length,
        "model_config": dict(model_config.__dict__),
        "run_config": _config_to_json(config),
        "metrics": metrics,
        "control_attribution": control_attribution,
        "collision_events": list(collision_events),
        "media": media,
        "runtime_s": round(time.time() - started_at, 3),
    }
    (output_dir / "carla_vision_closed_loop_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    _write_report_markdown(report, output_dir / "carla_vision_closed_loop_report.md")
    return report


def _load_bench2drive_planner(torch: Any, checkpoint_path: Path, device: Any) -> Tuple[Any, VisionE2EModelConfig]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model_config = VisionE2EModelConfig(**dict(checkpoint.get("model_config") or {}))
    model = _build_vision_e2e_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, model_config


def _can_connect_to_carla(carla: Any, host: str, port: int, *, timeout_s: float) -> bool:
    try:
        client = carla.Client(host, int(port))
        client.set_timeout(float(timeout_s))
        client.get_server_version()
        client.get_world().get_map().name
        return True
    except Exception:
        return False


def _launch_carla_server(config: CarlaVisionClosedLoopConfig) -> subprocess.Popen[Any]:
    carla_root = Path(config.carla_root).resolve()
    command = build_carla_launch_command(
        carla_root=carla_root,
        render_offscreen=True,
        null_rhi=False,
        port=int(config.port),
        quality_level=str(config.carla_quality_level or "Epic"),
        fps=int(config.fps),
    )
    command.extend(["-stdout", "-FullStdOutLogOutput"])
    if config.graphics_adapter >= 0:
        command.append(f"-graphicsadapter={config.graphics_adapter}")
    launch_script = carla_root / "CarlaUE4.sh"
    if not launch_script.exists():
        raise FileNotFoundError(f"CARLA launch script not found: {launch_script}")
    env = os.environ.copy()
    if config.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = str(config.cuda_visible_devices)
    log_path = Path(config.output_dir) / "carla_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab")
    try:
        process = subprocess.Popen(
            command,
            cwd=str(carla_root),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()
    setattr(process, "_nusc_scene_agent_log_path", log_path)
    (Path(config.output_dir) / "carla_server_process.json").write_text(
        json.dumps({"pid": process.pid, "command": command}), encoding="utf-8",
    )
    return process


def _resolve_carla_world(client: Any, town: str) -> Any:
    world = client.get_world()
    requested = str(town or "").strip()
    if not requested:
        return world
    current_name = str(world.get_map().name).rsplit("/", 1)[-1]
    if current_name == requested:
        return world
    return client.load_world(requested)


def _wait_for_carla(
    carla: Any,
    host: str,
    port: int,
    *,
    timeout_s: float,
    process: Optional[subprocess.Popen[Any]] = None,
) -> None:
    deadline = time.time() + float(timeout_s)
    last_error: Optional[Exception] = None
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            log_path = getattr(process, "_nusc_scene_agent_log_path", None)
            log_tail = _read_carla_log_tail(Path(log_path)) if log_path else ""
            detail = f"; log tail: {log_tail}" if log_tail else ""
            raise RuntimeError(
                f"CARLA server exited before becoming reachable "
                f"(returncode={process.returncode}){detail}"
            )
        try:
            client = carla.Client(host, int(port))
            client.set_timeout(5.0)
            client.get_server_version()
            client.get_world().get_map().name
            return
        except Exception as exc:
            last_error = exc
            time.sleep(2.0)
    raise RuntimeError(f"CARLA server did not become reachable on {host}:{port}: {last_error}")


def _terminate_carla_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except Exception:
        process.terminate()
    try:
        process.wait(timeout=15.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            process.kill()
        try:
            process.wait(timeout=5.0)
        except Exception:
            pass


def _read_carla_log_tail(path: Path, *, max_bytes: int = 4096) -> str:
    try:
        data = Path(path).read_bytes()[-int(max_bytes) :]
    except Exception:
        return ""
    return " ".join(data.decode("utf-8", errors="replace").split())


def _configure_carla_world(world: Any, config: CarlaVisionClosedLoopConfig) -> None:
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / max(int(config.fps), 1)
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    try:
        import carla  # type: ignore

        world.set_weather(carla.WeatherParameters.ClearNoon)
    except Exception:
        pass


def _configure_traffic_manager(client: Any, config: CarlaVisionClosedLoopConfig) -> Optional[Any]:
    try:
        traffic_manager = client.get_trafficmanager(int(config.traffic_manager_port))
    except Exception:
        return None
    try:
        traffic_manager.set_synchronous_mode(True)
    except Exception:
        pass
    try:
        traffic_manager.set_random_device_seed(int(config.seed))
    except Exception:
        pass
    try:
        traffic_manager.set_global_distance_to_leading_vehicle(3.0)
    except Exception:
        pass
    try:
        traffic_manager.set_hybrid_physics_mode(True)
        traffic_manager.set_hybrid_physics_radius(70.0)
    except Exception:
        pass
    try:
        traffic_manager.global_percentage_speed_difference(12.0)
    except Exception:
        pass
    return traffic_manager


def _condition_ego_route_traffic_lights(
    *,
    carla: Any,
    world: Any,
    ego: Any,
    route_points: Sequence[Mapping[str, float]],
    runtime: CarlaScenarioRuntime,
    enabled: bool,
) -> Dict[str, Any]:
    if not bool(enabled):
        runtime.metadata["traffic_light_conditioning"] = {
            "enabled": False,
            "selected_light_id": -1,
            "selected_light_ids": [],
            "group_size": 0,
        }
        return dict(runtime.metadata["traffic_light_conditioning"])
    selected = _select_ego_route_traffic_light(world=world, ego=ego, route_points=route_points)
    previous = dict(runtime.metadata.get("traffic_light_conditioning") or {})
    selected_history = [
        int(value)
        for value in list(previous.get("selected_light_ids") or [])
        if int(value) >= 0
    ]
    if selected is None:
        runtime.metadata["traffic_light_conditioning"] = {
            "enabled": True,
            "selected_light_id": -1,
            "selected_light_ids": selected_history,
            "group_size": 0,
        }
        return dict(runtime.metadata["traffic_light_conditioning"])
    group = _traffic_light_group(selected)
    selected_id = int(getattr(selected, "id", -1) or -1)
    if selected_id >= 0 and selected_id not in selected_history:
        selected_history.append(selected_id)
    for light in group:
        try:
            state = carla.TrafficLightState.Green if int(getattr(light, "id", -1) or -1) == selected_id else carla.TrafficLightState.Red
            light.set_state(state)
        except Exception:
            pass
        try:
            light.set_green_time(40.0 if int(getattr(light, "id", -1) or -1) == selected_id else 0.1)
            light.set_red_time(40.0 if int(getattr(light, "id", -1) or -1) != selected_id else 0.1)
            light.set_yellow_time(0.1)
        except Exception:
            pass
        try:
            light.freeze(True)
        except Exception:
            pass
    runtime.metadata["traffic_light_conditioning"] = {
        "enabled": True,
        "selected_light_id": selected_id,
        "selected_light_ids": selected_history,
        "group_size": len(group),
    }
    return dict(runtime.metadata["traffic_light_conditioning"])


def _select_ego_route_traffic_light(
    *,
    world: Any,
    ego: Any,
    route_points: Sequence[Mapping[str, float]],
) -> Optional[Any]:
    try:
        current = ego.get_traffic_light()
    except Exception:
        current = None
    if current is not None:
        return current
    if not route_points:
        return None
    try:
        ego_location = ego.get_location()
    except Exception:
        return None
    try:
        ego_progress, ego_lateral, _ = project_to_route((float(ego_location.x), float(ego_location.y)), route_points)
    except Exception:
        ego_progress, ego_lateral = 0.0, 0.0
    try:
        lights = list(world.get_actors().filter("traffic.traffic_light*"))
    except Exception:
        lights = []
    best: Optional[Tuple[float, Any]] = None
    for light in lights:
        try:
            location = light.get_location()
        except Exception:
            try:
                location = light.get_transform().location
            except Exception:
                continue
        progress, lateral, _ = project_to_route((float(location.x), float(location.y)), route_points)
        progress_gap = float(progress) - float(ego_progress)
        if progress_gap < -3.0 or progress_gap > 45.0:
            continue
        if abs(float(lateral) - float(ego_lateral)) > 16.0:
            continue
        score = abs(progress_gap - 12.0) + 0.35 * abs(float(lateral) - float(ego_lateral))
        if best is None or score < float(best[0]):
            best = (score, light)
    return None if best is None else best[1]


def _traffic_light_group(light: Any) -> List[Any]:
    try:
        group = list(light.get_group_traffic_lights() or [])
    except Exception:
        group = []
    if not group:
        group = [light]
    seen = set()
    unique = []
    for item in group:
        key = int(getattr(item, "id", id(item)) or id(item))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _traffic_manager_port(traffic_manager: Optional[Any]) -> int:
    if traffic_manager is None:
        return 8000
    try:
        return int(traffic_manager.get_port())
    except Exception:
        return 8000


def _enable_traffic_manager_vehicle(
    *,
    actor: Any,
    traffic_manager: Optional[Any],
    speed_mps: float,
    auto_lane_change: bool = False,
) -> bool:
    if traffic_manager is None:
        return False
    try:
        actor.set_autopilot(True, _traffic_manager_port(traffic_manager))
    except Exception:
        return False
    try:
        traffic_manager.auto_lane_change(actor, bool(auto_lane_change))
    except Exception:
        pass
    _set_traffic_manager_speed(traffic_manager=traffic_manager, actor=actor, speed_mps=speed_mps)
    return True


def _set_traffic_manager_speed(*, traffic_manager: Optional[Any], actor: Any, speed_mps: float) -> bool:
    if traffic_manager is None:
        return False
    target_kmh = max(float(speed_mps), 0.0) * 3.6
    try:
        traffic_manager.set_desired_speed(actor, float(target_kmh))
        return True
    except Exception:
        pass
    try:
        speed_limit_kmh = max(float(actor.get_speed_limit()), 1.0)
    except Exception:
        speed_limit_kmh = 50.0
    percentage = 100.0 - target_kmh / max(speed_limit_kmh, 1.0) * 100.0
    percentage = max(min(percentage, 95.0), -45.0)
    try:
        traffic_manager.vehicle_percentage_speed_difference(actor, float(percentage))
        return True
    except Exception:
        return False


def _select_carla_route(
    *,
    carla: Any,
    carla_map: Any,
    spawn_index: int,
    destination_index: int,
    sampling_resolution_m: float,
    min_route_length_m: float = MIN_DEFAULT_ROUTE_LENGTH_M,
    max_route_length_m: float = MAX_DEFAULT_ROUTE_LENGTH_M,
    preferred_route_length_m: float = PREFERRED_DEFAULT_ROUTE_LENGTH_M,
    required_commands: Optional[Sequence[int]] = None,
    require_crosswalk_near_command: bool = False,
    crosswalk_command_offset_m: float = 8.0,
    crosswalk_search_window_m: float = 28.0,
    crosswalk_max_route_distance_m: float = 9.0,
    search_start_candidates: bool = False,
    start_candidate_count: int = 1,
    start_candidate_radius_m: float = 0.0,
    maneuver_min_progress_m: float = 0.0,
    maneuver_max_progress_m: float = 0.0,
    min_post_maneuver_length_m: float = 0.0,
    disallow_lane_change_before_maneuver: bool = False,
    prefer_rightmost_start_lane: bool = False,
    require_rightmost_start_lane: bool = False,
) -> Tuple[List[Any], List[Dict[str, float]], Any, Any]:
    from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore

    spawn_points = list(carla_map.get_spawn_points())
    if len(spawn_points) < 2:
        raise RuntimeError("CARLA map contains fewer than two spawn points.")
    planner = GlobalRoutePlanner(carla_map, float(sampling_resolution_m))
    start_idx = int(spawn_index) % len(spawn_points)
    errors = []
    required_command_set = {int(value) for value in list(required_commands or [])}
    route_candidates: List[Tuple[float, List[Any], List[Dict[str, float]], Any, Any]] = []
    start_candidates = _route_start_candidates(
        carla=carla,
        carla_map=carla_map,
        spawn_points=spawn_points,
        start_idx=start_idx,
        search_start_candidates=bool(search_start_candidates),
        start_candidate_count=int(start_candidate_count),
        start_candidate_radius_m=float(start_candidate_radius_m),
        prefer_rightmost_start_lane=bool(prefer_rightmost_start_lane),
        require_rightmost_start_lane=bool(require_rightmost_start_lane),
    )
    for start_rank, start_penalty, start_index, start in start_candidates:
        candidates: List[Tuple[float, int, Any]] = []
        if destination_index >= 0:
            dest = spawn_points[int(destination_index) % len(spawn_points)]
            candidates.append((start.location.distance(dest.location), int(destination_index) % len(spawn_points), dest))
        else:
            for idx, transform in enumerate(spawn_points):
                if idx == start_index:
                    continue
                candidates.append((start.location.distance(transform.location), idx, transform))
            candidates.sort(key=lambda item: item[0])
        for _, _, dest in candidates:
            try:
                trace = planner.trace_route(start.location, dest.location)
                route_points = _route_trace_to_points(trace)
                route_length = _route_length(route_points)
                command_progress = _route_command_progresses(route_points, required_command_set)
                command_segments = _route_command_segments(route_points, required_command_set)
                first_maneuver_start = command_segments[0][0] if command_segments else (min(command_progress) if command_progress else 0.0)
                first_maneuver_end = command_segments[0][1] if command_segments else first_maneuver_start
                if required_command_set and not command_progress:
                    continue
                if command_segments:
                    if float(maneuver_min_progress_m) > 0.0 and first_maneuver_start < float(maneuver_min_progress_m):
                        continue
                    if float(maneuver_max_progress_m) > 0.0 and first_maneuver_start > float(maneuver_max_progress_m):
                        continue
                    if float(min_post_maneuver_length_m) > 0.0 and route_length - first_maneuver_end < float(min_post_maneuver_length_m):
                        continue
                    lane_changes_before = _route_lane_change_progresses(route_points, before_progress_m=first_maneuver_start)
                    if bool(disallow_lane_change_before_maneuver) and lane_changes_before:
                        continue
                else:
                    lane_changes_before = []
                crosswalk_progress = None
                if require_crosswalk_near_command:
                    preferred_progresses = [
                        progress + float(crosswalk_command_offset_m)
                        for progress in (command_progress or [route_length * 0.35])
                    ]
                    crosswalk_progress = _nearest_crosswalk_progress(
                        carla_map=carla_map,
                        route_points=route_points,
                        preferred_progresses=preferred_progresses,
                        search_window_m=float(crosswalk_search_window_m),
                        max_route_distance_m=float(crosswalk_max_route_distance_m),
                    )
                    if crosswalk_progress is None:
                        continue
                if (
                    destination_index >= 0
                    and len(route_points) >= 2
                    and route_length >= 1.0
                    and not required_command_set
                    and not require_crosswalk_near_command
                    and not search_start_candidates
                ):
                    return trace, route_points, start, dest
                if len(route_points) >= 20 and float(min_route_length_m) <= route_length <= float(max_route_length_m):
                    preferred = float(preferred_route_length_m or 0.0)
                    length_score = abs(route_length - preferred) if preferred > 0.0 else route_length
                    command_score = first_maneuver_start if command_segments else (min(command_progress) if command_progress else route_length * 0.5)
                    crosswalk_score = float(crosswalk_progress) if crosswalk_progress is not None else command_score
                    approach_target = 0.5 * (
                        float(maneuver_min_progress_m or 0.0) + float(maneuver_max_progress_m or 0.0)
                    )
                    approach_score = abs(command_score - approach_target) if approach_target > 0.0 else 0.0
                    lane_change_penalty = 18.0 * len(lane_changes_before)
                    score = (
                        length_score
                        + 0.08 * command_score
                        + 0.18 * approach_score
                        + 0.35 * abs(crosswalk_score - command_score)
                        + lane_change_penalty
                        + float(start_penalty)
                        + 0.05 * float(start_rank)
                    )
                    route_candidates.append((score, trace, route_points, start, dest))
            except Exception as exc:
                errors.append(str(exc))
    if route_candidates:
        route_candidates.sort(key=lambda item: item[0])
        _, trace, route_points, start, dest = route_candidates[0]
        return trace, route_points, start, dest
    raise RuntimeError(f"Unable to build a CARLA route from spawn index {spawn_index}. Errors: {errors[:3]}")


def _route_trace_to_points(route_trace: Sequence[Any]) -> List[Dict[str, float]]:
    points = []
    for waypoint, road_option in route_trace:
        location = waypoint.transform.location
        rotation = waypoint.transform.rotation
        points.append(
            {
                "x": float(location.x),
                "y": float(location.y),
                "z": float(location.z),
                "yaw": math.radians(float(rotation.yaw)),
                "command": float(int(road_option)),
            }
        )
    return points


def _route_start_candidates(
    *,
    carla: Any,
    carla_map: Any,
    spawn_points: Sequence[Any],
    start_idx: int,
    search_start_candidates: bool,
    start_candidate_count: int,
    start_candidate_radius_m: float,
    prefer_rightmost_start_lane: bool,
    require_rightmost_start_lane: bool,
) -> List[Tuple[float, float, int, Any]]:
    if not spawn_points:
        return []
    start_idx = int(start_idx) % len(spawn_points)
    base = spawn_points[start_idx]
    if not search_start_candidates:
        return [(0.0, 0.0, start_idx, base)]

    candidates: List[Tuple[float, float, int, Any]] = []
    for idx, transform in enumerate(spawn_points):
        try:
            distance = float(base.location.distance(transform.location))
        except Exception:
            distance = 0.0 if idx == start_idx else float("inf")
        if idx != start_idx and float(start_candidate_radius_m) > 0.0 and distance > float(start_candidate_radius_m):
            continue
        lane_penalty = _rightmost_start_lane_penalty(carla=carla, carla_map=carla_map, transform=transform)
        if bool(require_rightmost_start_lane) and lane_penalty > 1e-6:
            continue
        if not bool(prefer_rightmost_start_lane):
            lane_penalty = 0.0
        distance_penalty = 0.04 * min(distance, 80.0)
        candidates.append((float(lane_penalty + distance_penalty), idx, transform))
    if not candidates and not bool(require_rightmost_start_lane):
        candidates.append((0.0, start_idx, base))
    candidates.sort(key=lambda item: (item[0], abs(int(item[1]) - start_idx)))
    selected = candidates[: max(int(start_candidate_count), 1)]
    return [(float(rank), float(score), int(idx), transform) for rank, (score, idx, transform) in enumerate(selected)]


def _rightmost_start_lane_penalty(*, carla: Any, carla_map: Any, transform: Any) -> float:
    del carla
    try:
        waypoint = carla_map.get_waypoint(
            transform.location,
            project_to_road=True,
            lane_type=1,
        )
    except Exception:
        waypoint = None
    if waypoint is None:
        return 8.0
    try:
        lane_id = int(getattr(waypoint, "lane_id", 0) or 0)
    except Exception:
        lane_id = 0
    right_lanes = 0
    queue = [waypoint]
    seen = set()
    for _ in range(16):
        if not queue:
            break
        current = queue.pop(0)
        try:
            current_key = (
                int(getattr(current, "road_id", 0) or 0),
                int(getattr(current, "section_id", 0) or 0),
                int(getattr(current, "lane_id", 0) or 0),
            )
        except Exception:
            current_key = id(current)
        if current_key in seen:
            continue
        seen.add(current_key)
        if current is not waypoint:
            try:
                adjacent_lane_id = int(getattr(current, "lane_id", 0) or 0)
            except Exception:
                adjacent_lane_id = 0
            same_side = lane_id == 0 or adjacent_lane_id == 0 or (lane_id > 0) == (adjacent_lane_id > 0)
            outward = abs(adjacent_lane_id) > abs(lane_id)
            if _is_driving_lane_waypoint(current) and same_side and outward:
                right_lanes += 1
        try:
            left_lane = current.get_left_lane()
        except Exception:
            left_lane = None
        try:
            right_lane = current.get_right_lane()
        except Exception:
            right_lane = None
        for neighbor in (left_lane, right_lane):
            if neighbor is not None:
                queue.append(neighbor)
    return float(6.0 * right_lanes)


def _is_driving_lane_waypoint(waypoint: Any) -> bool:
    try:
        lane_type = getattr(waypoint, "lane_type", "")
    except Exception:
        return False
    text = str(lane_type).lower()
    if "driving" in text:
        return True
    try:
        return int(lane_type) == 1
    except Exception:
        return False


def _route_command_ids_from_maneuver(value: Any) -> List[int]:
    text = str(value or "").strip().lower()
    if not text:
        return []
    aliases = {
        "left": [1],
        "right": [2],
        "straight": [3],
        "lane_follow": [4],
        "lanefollow": [4],
        "change_lane_left": [5],
        "changelaneleft": [5],
        "change_lane_right": [6],
        "changelaneright": [6],
    }
    if text in aliases:
        return list(aliases[text])
    result = []
    for token in text.replace(",", " ").split():
        try:
            result.append(int(token))
        except ValueError:
            continue
    return result


def _route_command_progresses(route_points: Sequence[Mapping[str, float]], command_ids: set[int]) -> List[float]:
    if not route_points or not command_ids:
        return []
    progresses = []
    cumulative = 0.0
    previous: Optional[Mapping[str, float]] = None
    for point in route_points:
        if previous is not None:
            cumulative += math.hypot(float(point["x"]) - float(previous["x"]), float(point["y"]) - float(previous["y"]))
        if int(float(point.get("command") or 4.0)) in command_ids:
            progresses.append(float(cumulative))
        previous = point
    return progresses


def _route_command_segments(route_points: Sequence[Mapping[str, float]], command_ids: set[int]) -> List[Tuple[float, float]]:
    if not route_points or not command_ids:
        return []
    segments: List[Tuple[float, float]] = []
    cumulative = 0.0
    current_start: Optional[float] = None
    previous: Optional[Mapping[str, float]] = None
    for point in route_points:
        if previous is not None:
            cumulative += math.hypot(float(point["x"]) - float(previous["x"]), float(point["y"]) - float(previous["y"]))
        command = int(float(point.get("command") or 4.0))
        if command in command_ids and current_start is None:
            current_start = float(cumulative)
        if command not in command_ids and current_start is not None:
            segments.append((float(current_start), float(cumulative)))
            current_start = None
        previous = point
    if current_start is not None:
        segments.append((float(current_start), float(cumulative)))
    return segments


def _route_lane_change_progresses(route_points: Sequence[Mapping[str, float]], *, before_progress_m: float) -> List[float]:
    if not route_points:
        return []
    progresses: List[float] = []
    cumulative = 0.0
    previous: Optional[Mapping[str, float]] = None
    for point in route_points:
        if previous is not None:
            cumulative += math.hypot(float(point["x"]) - float(previous["x"]), float(point["y"]) - float(previous["y"]))
        if cumulative >= float(before_progress_m):
            break
        if int(float(point.get("command") or 4.0)) in {5, 6}:
            progresses.append(float(cumulative))
        previous = point
    return progresses


def _nearest_crosswalk_progress(
    *,
    carla_map: Any,
    route_points: Sequence[Mapping[str, float]],
    preferred_progresses: Sequence[float],
    search_window_m: float,
    max_route_distance_m: float,
) -> Optional[float]:
    if not route_points or not preferred_progresses:
        return None
    try:
        crosswalk_locations = list(carla_map.get_crosswalks() or [])
    except Exception:
        crosswalk_locations = []
    best: Optional[Tuple[float, float]] = None
    for location in crosswalk_locations:
        progress, lateral, _ = project_to_route((float(location.x), float(location.y)), route_points)
        if abs(float(lateral)) > max(float(max_route_distance_m), 1.0):
            continue
        progress_error = min(abs(float(progress) - float(preferred)) for preferred in preferred_progresses)
        if progress_error > max(float(search_window_m), 1.0):
            continue
        score = float(progress_error) + 0.25 * abs(float(lateral))
        if best is None or score < best[0]:
            best = (score, float(progress))
    return None if best is None else best[1]


def _route_constraints_from_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    maneuver = (
        params.get("route_maneuver")
        or params.get("required_route_maneuver")
        or params.get("required_maneuver")
    )
    required_commands = _route_command_ids_from_maneuver(maneuver)
    return {
        "required_commands": required_commands,
        "require_crosswalk_near_command": _scenario_bool(params, "require_crosswalk_near_maneuver", False),
        "crosswalk_command_offset_m": _scenario_float(params, "crosswalk_maneuver_offset_m", 8.0),
        "crosswalk_search_window_m": _scenario_float(params, "crosswalk_search_window_m", 28.0),
        "crosswalk_max_route_distance_m": _scenario_float(params, "crosswalk_max_route_distance_m", 9.0),
        "search_start_candidates": _scenario_bool(params, "search_start_candidates", False),
        "start_candidate_count": int(_scenario_float(params, "start_candidate_count", 1.0)),
        "start_candidate_radius_m": _scenario_float(params, "start_candidate_radius_m", 0.0),
        "maneuver_min_progress_m": _scenario_float(params, "maneuver_min_progress_m", 0.0),
        "maneuver_max_progress_m": _scenario_float(params, "maneuver_max_progress_m", 0.0),
        "min_post_maneuver_length_m": _scenario_float(params, "min_post_maneuver_length_m", 0.0),
        "disallow_lane_change_before_maneuver": _scenario_bool(params, "disallow_lane_change_before_maneuver", False),
        "prefer_rightmost_start_lane": _scenario_bool(params, "prefer_rightmost_start_lane", False),
        "require_rightmost_start_lane": _scenario_bool(params, "require_rightmost_start_lane", False),
    }


def _spawn_ego_vehicle(*, carla: Any, world: Any, spawn_transform: Any) -> Any:
    blueprints = world.get_blueprint_library()
    vehicle_bp = _first_blueprint(
        blueprints,
        ["vehicle.tesla.model3", "vehicle.lincoln.mkz_2020", "vehicle.audi.tt"],
    )
    vehicle_bp.set_attribute("role_name", "hero")
    if vehicle_bp.has_attribute("color"):
        vehicle_bp.set_attribute("color", "32,96,160")
    for attempt in range(8):
        transform = carla.Transform(
            spawn_transform.location + carla.Location(z=0.15 + 0.05 * attempt),
            spawn_transform.rotation,
        )
        actor = world.try_spawn_actor(vehicle_bp, transform)
        if actor is not None:
            return actor
    raise RuntimeError("Unable to spawn ego vehicle at the selected CARLA spawn point.")


def _first_blueprint(blueprints: Any, names: Sequence[str]) -> Any:
    for name in names:
        matches = blueprints.filter(name)
        if matches:
            return matches[0]
    all_vehicles = blueprints.filter("vehicle.*")
    if not all_vehicles:
        raise RuntimeError("No vehicle blueprint found in CARLA.")
    return all_vehicles[0]


def _normalize_scenario_specs(scenarios: Optional[Sequence[Mapping[str, Any]]]) -> List[CarlaVisionScenarioSpec]:
    raw_specs = list(scenarios or DEFAULT_CARLA_VISION_SCENARIOS)
    specs = []
    standard_keys = {"name", "scenario_type", "type", "spawn_index", "destination_index", "target_speed_mps", "parameters", "params"}
    for idx, raw in enumerate(raw_specs):
        scenario_type = str(raw.get("scenario_type") or raw.get("type") or raw.get("name") or "free_drive")
        name = str(raw.get("name") or scenario_type or f"scenario_{idx}")
        target_speed = raw.get("target_speed_mps")
        parameters = dict(raw.get("parameters") or raw.get("params") or {})
        parameters.update({str(key): value for key, value in raw.items() if key not in standard_keys})
        specs.append(
            CarlaVisionScenarioSpec(
                name=name,
                scenario_type=scenario_type,
                spawn_index=int(raw.get("spawn_index") or 0),
                destination_index=int(raw.get("destination_index") if raw.get("destination_index") is not None else -1),
                target_speed_mps=None if target_speed is None else float(target_speed),
                parameters=parameters,
            )
        )
    return specs


def _safe_filename(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip().lower())
    return text.strip("_") or "scenario"


def _spawn_scenario_runtime(
    *,
    carla: Any,
    client: Any,
    world: Any,
    carla_map: Any,
    traffic_manager: Optional[Any],
    route_points: Sequence[Mapping[str, float]],
    config: CarlaVisionClosedLoopConfig,
) -> CarlaScenarioRuntime:
    scenario_type = str(config.scenario_type or "free_drive")
    name = str(config.scenario_name or scenario_type)
    params = dict(config.scenario_params or {})
    if scenario_type in {"free_drive", "none", "route_following"}:
        return CarlaScenarioRuntime(
            name=name,
            scenario_type="free_drive",
            actors=[],
            metadata={"description": "navigation route rollout"},
        )
    if scenario_type == "pedestrian_crossing":
        progress_m = _resolve_pedestrian_crossing_progress(
            carla_map=carla_map,
            route_points=route_points,
            params=params,
            default_progress_m=20.0,
        )
        lateral_offset_m = _scenario_float(params, "lateral_offset_m", -4.0)
        actor, metadata = _spawn_crossing_walker(
            carla=carla,
            world=world,
            carla_map=carla_map,
            route_points=route_points,
            progress_m=progress_m,
            lateral_offset_m=lateral_offset_m,
            speed_mps=_scenario_float(params, "actor_speed_mps", _scenario_float(params, "walker_speed_mps", 1.4)),
            search_window_m=_scenario_float(params, "crosswalk_search_window_m", 35.0),
            max_route_distance_m=_scenario_float(params, "crosswalk_max_route_distance_m", 10.0),
            allow_route_fallback=not _scenario_bool(params, "require_crosswalk", True),
            candidate_offset=int(_scenario_float(params, "crosswalk_candidate_offset", 0.0)),
        )
        metadata.update(
            {
                "trigger_progress_m": _scenario_float(
                    params,
                    "trigger_progress_m",
                    max(
                        float(metadata.get("progress_m") or progress_m)
                        - _scenario_float(params, "trigger_distance_before_crossing_m", 14.0),
                        0.0,
                    ),
                ),
                "event_end_progress_m": _scenario_float(
                    params,
                    "event_end_progress_m",
                    float(metadata.get("progress_m") or progress_m)
                    + _scenario_float(params, "event_end_after_crossing_m", 42.0),
                ),
                "pedestrian_trigger_mode": str(params.get("pedestrian_trigger_mode") or "ego_progress"),
                "pedestrian_trigger_time_s": _scenario_float(params, "pedestrian_trigger_time_s", 0.0),
                "pedestrian_trigger_buffer_m": _scenario_float(params, "pedestrian_trigger_buffer_m", 0.0),
                "route_maneuver": str(params.get("route_maneuver") or ""),
                "required_route_commands": _route_command_ids_from_maneuver(params.get("route_maneuver")),
                "ambient_safety_mode": str(params.get("ambient_safety_mode") or "emergency_only").strip().lower(),
                "fps": int(config.fps),
            }
        )
        actors = [actor]
        actor_specs: List[Dict[str, Any]] = []
        pedestrian_count = max(int(_scenario_float(params, "pedestrian_count", 4.0)), 1)
        second_wave_count = max(int(_scenario_float(params, "pedestrian_second_wave_count", 0.0)), 0)
        group_spacing_m = _scenario_float(params, "pedestrian_group_spacing_m", 0.85)
        group_depth_m = _scenario_float(params, "pedestrian_group_depth_m", 0.55)
        start_delay_step_s = _scenario_float(params, "pedestrian_start_delay_step_s", 0.22)
        second_wave_delay_s = max(_scenario_float(params, "pedestrian_second_wave_delay_s", 2.8), 0.0)
        start_advance_s = abs(_scenario_float(params, "pedestrian_start_advance_s", 0.8))
        total_pedestrian_count = pedestrian_count + second_wave_count
        actor_specs.append(
            _crossing_walker_spec(
                actor_index=0,
                metadata=metadata,
                progress_m=progress_m,
                lateral_offset_m=lateral_offset_m,
                order=0,
                speed_scale=1.0,
                start_delay_s=-start_advance_s,
            )
        )
        for member_actor, member_metadata in _spawn_crossing_walker_group_members(
            carla=carla,
            world=world,
            base_metadata=metadata,
            start_index=1,
            count=total_pedestrian_count - 1,
            group_spacing_m=group_spacing_m,
            group_depth_m=group_depth_m,
        ):
            actors.append(member_actor)
            actor_index = len(actors) - 1
            order = int(member_metadata.get("group_order") or actor_index)
            wave_delay_s = second_wave_delay_s if order >= pedestrian_count else 0.0
            actor_specs.append(
                _crossing_walker_spec(
                    actor_index=actor_index,
                    metadata={**metadata, **member_metadata},
                    progress_m=progress_m,
                    lateral_offset_m=lateral_offset_m,
                    order=order,
                    speed_scale=0.96 + 0.025 * (order % 4),
                    start_delay_s=-start_advance_s + wave_delay_s + float(order) * float(start_delay_step_s),
                )
            )
        for traffic_actor, traffic_metadata in _spawn_ambient_autopilot_traffic(
            carla=carla,
            client=client,
            world=world,
            carla_map=carla_map,
            traffic_manager=traffic_manager,
            route_points=route_points,
            max_count=int(_scenario_float(params, "ambient_vehicle_count", 12.0)),
            max_lateral_m=_scenario_float(params, "ambient_max_lateral_m", 28.0),
            target_speed_mps=_scenario_float(params, "ambient_target_speed_mps", 5.6),
            speed_variation_mps=_scenario_float(params, "ambient_speed_variation_mps", 1.2),
            lane_change_percentage=_scenario_float(params, "ambient_lane_change_percentage", 20.0),
            preferred_progress_m=_optional_scenario_float(params, "ambient_preferred_progress_m"),
            min_progress_m=_scenario_float(params, "ambient_min_progress_m", 18.0),
            max_progress_m=_optional_scenario_float(params, "ambient_max_progress_m"),
            occupied_progresses=[float(metadata.get("progress_m") or progress_m)],
            excluded_progress_ranges=[
                (
                    max(float(metadata.get("progress_m") or progress_m) - 9.0, 0.0),
                    float(metadata.get("progress_m") or progress_m) + 16.0,
                )
            ],
        ):
            actors.append(traffic_actor)
            actor_specs.append({"actor_index": len(actors) - 1, **traffic_metadata})
        metadata["pedestrian_count"] = len([spec for spec in actor_specs if str(spec.get("kind") or "") == "walker"])
        metadata["pedestrian_generation"] = "crosswalk_group"
        metadata["actor_specs"] = actor_specs
        return CarlaScenarioRuntime(name=name, scenario_type=scenario_type, actors=actors, metadata=metadata)
    if scenario_type == "dense_follow_overtake":
        return _spawn_natural_vehicle_traffic_runtime(
            carla=carla,
            client=client,
            world=world,
            carla_map=carla_map,
            traffic_manager=traffic_manager,
            route_points=route_points,
            name=name,
            params=params,
            scenario_type="dense_follow_overtake",
            event="natural_dense_traffic",
            description="short natural-traffic follow and passing-opportunity scene",
            default_vehicle_count=28,
            default_target_speed_mps=5.4,
            default_lane_change_percentage=25.0,
        )
    if scenario_type == "adjacent_lane_cut_in":
        return _spawn_natural_vehicle_traffic_runtime(
            carla=carla,
            client=client,
            world=world,
            carla_map=carla_map,
            traffic_manager=traffic_manager,
            route_points=route_points,
            name=name,
            params=params,
            scenario_type="adjacent_lane_cut_in",
            event="natural_adjacent_lane_interaction",
            description="short natural-traffic adjacent-lane interaction scene",
            default_vehicle_count=24,
            default_target_speed_mps=5.2,
            default_lane_change_percentage=55.0,
        )
    raise ValueError(f"Unknown CARLA vision scenario type: {scenario_type}")


def _scenario_float(params: Mapping[str, Any], key: str, default: float) -> float:
    value = params.get(key)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _scenario_bool(params: Mapping[str, Any], key: str, default: bool) -> bool:
    value = params.get(key)
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _optional_scenario_float(params: Mapping[str, Any], key: str) -> Optional[float]:
    value = params.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_pedestrian_crossing_progress(
    *,
    carla_map: Any,
    route_points: Sequence[Mapping[str, float]],
    params: Mapping[str, Any],
    default_progress_m: float,
) -> float:
    raw_progress = params.get("progress_m")
    if raw_progress is not None and str(raw_progress).strip().lower() not in {"", "auto", "right_turn_crosswalk"}:
        try:
            return float(raw_progress)
        except (TypeError, ValueError):
            pass
    command_ids = _route_command_ids_from_maneuver(params.get("route_maneuver") or params.get("required_maneuver"))
    command_progresses = _route_command_progresses(route_points, set(command_ids))
    if command_progresses:
        preferred = [
            float(progress) + _scenario_float(params, "crosswalk_maneuver_offset_m", 8.0)
            for progress in command_progresses
        ]
        crosswalk_progress = _nearest_crosswalk_progress(
            carla_map=carla_map,
            route_points=route_points,
            preferred_progresses=preferred,
            search_window_m=_scenario_float(params, "crosswalk_search_window_m", 28.0),
            max_route_distance_m=_scenario_float(params, "crosswalk_max_route_distance_m", 9.0),
        )
        if crosswalk_progress is not None:
            return float(crosswalk_progress)
        return float(preferred[0])
    return float(default_progress_m)


def _spawn_natural_vehicle_traffic_runtime(
    *,
    carla: Any,
    client: Any,
    world: Any,
    carla_map: Any,
    traffic_manager: Optional[Any],
    route_points: Sequence[Mapping[str, float]],
    name: str,
    params: Mapping[str, Any],
    scenario_type: str,
    event: str,
    description: str,
    default_vehicle_count: int,
    default_target_speed_mps: float,
    default_lane_change_percentage: float,
) -> CarlaScenarioRuntime:
    actors: List[Any] = []
    actor_specs: List[Dict[str, Any]] = []
    for actor, metadata in _spawn_ambient_autopilot_traffic(
        carla=carla,
        client=client,
        world=world,
        carla_map=carla_map,
        traffic_manager=traffic_manager,
        route_points=route_points,
        max_count=int(_scenario_float(params, "ambient_vehicle_count", float(default_vehicle_count))),
        max_lateral_m=_scenario_float(params, "ambient_max_lateral_m", 35.0),
        target_speed_mps=_scenario_float(params, "ambient_target_speed_mps", default_target_speed_mps),
        speed_variation_mps=_scenario_float(params, "ambient_speed_variation_mps", 1.4),
        lane_change_percentage=_scenario_float(
            params,
            "ambient_lane_change_percentage",
            default_lane_change_percentage,
        ),
        preferred_progress_m=_optional_scenario_float(params, "ambient_preferred_progress_m"),
        min_progress_m=_scenario_float(params, "ambient_min_progress_m", 18.0),
        max_progress_m=_optional_scenario_float(params, "ambient_max_progress_m"),
    ):
        actors.append(actor)
        actor_specs.append({"actor_index": len(actors) - 1, **metadata})
    return CarlaScenarioRuntime(
        name=name,
        scenario_type=str(scenario_type),
        actors=actors,
        metadata={
            "description": description,
            "vehicle_generation": "traffic_manager_autopilot",
            "actor_specs": actor_specs,
            "passive_events": [{"event": event, "event_start_progress_m": 0.0, "event_end_progress_m": _route_length(route_points)}],
            "active_event": "initializing",
            "active_events": ["initializing"],
        },
    )


def _spawn_ambient_autopilot_traffic(
    *,
    carla: Any,
    client: Any,
    world: Any,
    carla_map: Any,
    traffic_manager: Optional[Any],
    route_points: Sequence[Mapping[str, float]],
    max_count: int,
    max_lateral_m: float = 35.0,
    target_speed_mps: float = 5.0,
    speed_variation_mps: float = 0.0,
    lane_change_percentage: float = 0.0,
    preferred_progress_m: Optional[float] = None,
    min_progress_m: float = 18.0,
    max_progress_m: Optional[float] = None,
    occupied_progresses: Optional[Sequence[float]] = None,
    excluded_progress_ranges: Optional[Sequence[Tuple[float, float]]] = None,
) -> List[Tuple[Any, Dict[str, Any]]]:
    if traffic_manager is None or max_count <= 0:
        return []
    blueprints = world.get_blueprint_library()
    vehicle_bps = _ambient_vehicle_blueprints(blueprints)
    if not vehicle_bps:
        return []
    try:
        from carla.command import FutureActor, SetAutopilot, SpawnActor  # type: ignore
    except Exception:
        return []

    spawn_points = list(carla_map.get_spawn_points())
    route_length = _route_length(route_points)
    min_progress = max(float(min_progress_m), 0.0)
    max_progress = route_length - 6.0 if max_progress_m is None else min(float(max_progress_m), route_length - 6.0)
    max_progress = max(max_progress, min_progress)
    preferred_progress = (
        max(min(float(preferred_progress_m), max_progress), min_progress)
        if preferred_progress_m is not None
        else route_length * 0.52
    )
    occupied = [float(value) for value in list(occupied_progresses or [])]
    scored = []
    for spawn_index, transform in enumerate(spawn_points):
        progress, lateral, _ = project_to_route(
            (float(transform.location.x), float(transform.location.y)),
            route_points,
        )
        if progress < min_progress or progress > max_progress:
            continue
        if any(float(start) <= float(progress) <= float(end) for start, end in list(excluded_progress_ranges or [])):
            continue
        if abs(lateral) < 4.2 and any(abs(float(progress) - used) < 12.0 for used in occupied):
            continue
        if abs(lateral) > max(float(max_lateral_m), 1.0):
            continue
        # Favor traffic that will appear in the camera near intersections and route-adjacent lanes,
        # while avoiding the ego spawn area and reserved event actors.
        route_neighbor_score = 0.45 * min(abs(lateral), 20.0)
        sequence_score = 0.02 * abs(progress - preferred_progress)
        scored.append((route_neighbor_score + sequence_score, spawn_index, progress, lateral, transform))
    scored.sort(key=lambda item: item[0])
    selected = []
    used_spots: List[Tuple[float, float]] = []
    for _, spawn_index, progress, lateral, transform in scored:
        if len(selected) >= int(max_count):
            break
        if any(
            abs(progress - used_progress) < 5.0 and abs(lateral - used_lateral) < 2.5
            for used_progress, used_lateral in used_spots
        ):
            continue
        used_spots.append((float(progress), float(lateral)))
        selected.append((spawn_index, progress, lateral, transform))

    batch = []
    metadata_by_order: List[Dict[str, Any]] = []
    tm_port = _traffic_manager_port(traffic_manager)
    for order, (spawn_index, progress, lateral, transform) in enumerate(selected):
        bp = vehicle_bps[order % len(vehicle_bps)]
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "autopilot")
        if bp.has_attribute("color"):
            colors = bp.get_attribute("color").recommended_values
            if colors:
                bp.set_attribute("color", colors[order % len(colors)])
        target_speed = max(float(target_speed_mps) + ((order % 5) - 2) * float(speed_variation_mps) * 0.25, 1.0)
        batch.append(SpawnActor(bp, transform).then(SetAutopilot(FutureActor, True, tm_port)))
        metadata_by_order.append(
            {
                "role": "ambient_autopilot_vehicle",
                "kind": "autopilot_vehicle",
                "event": "ambient_traffic",
                "placement": "carla_spawn_point",
                "spawn_point_index": int(spawn_index),
                "progress_m": float(progress),
                "route_lateral_m": float(lateral),
                "target_speed_mps": float(target_speed),
                "control_mode": "traffic_manager",
                "event_start_progress_m": max(float(progress) - 30.0, 0.0),
                "event_end_progress_m": min(float(progress) + 30.0, route_length),
                "lane_change_percentage": float(lane_change_percentage),
            }
        )
    if not batch:
        return []

    responses = client.apply_batch_sync(batch, True)
    actor_ids = []
    kept_metadata = []
    for response, metadata in zip(responses, metadata_by_order):
        if getattr(response, "error", None):
            continue
        actor_ids.append(int(response.actor_id))
        kept_metadata.append(metadata)

    actor_lookup = {int(actor.id): actor for actor in world.get_actors(actor_ids)}
    actors: List[Tuple[Any, Dict[str, Any]]] = []
    for actor_id, metadata in zip(actor_ids, kept_metadata):
        actor = actor_lookup.get(int(actor_id))
        if actor is None:
            continue
        try:
            traffic_manager.auto_lane_change(actor, True)
        except Exception:
            pass
        try:
            traffic_manager.random_left_lanechange_percentage(actor, float(lane_change_percentage))
            traffic_manager.random_right_lanechange_percentage(actor, float(lane_change_percentage))
        except Exception:
            pass
        try:
            _set_traffic_manager_speed(
                traffic_manager=traffic_manager,
                actor=actor,
                speed_mps=float(metadata.get("target_speed_mps") or target_speed_mps),
            )
        except Exception:
            pass
        try:
            traffic_manager.vehicle_percentage_speed_difference(actor, 10.0 + (len(actors) % 4) * 4.0)
        except Exception:
            pass
        actors.append((actor, {**metadata, "actor_type": str(getattr(actor, "type_id", ""))}))
    return actors


def _ambient_vehicle_blueprints(blueprints: Any) -> List[Any]:
    preferred = [
        "vehicle.tesla.model3",
        "vehicle.lincoln.mkz_2020",
        "vehicle.audi.tt",
        "vehicle.dodge.charger_2020",
        "vehicle.mini.cooper_s",
        "vehicle.audi.etron",
        "vehicle.mercedes.coupe",
        "vehicle.toyota.prius",
    ]
    result = []
    for name in preferred:
        matches = blueprints.filter(name)
        if matches:
            result.append(matches[0])
    return result


def _spawn_crossing_walker(
    *,
    carla: Any,
    world: Any,
    carla_map: Any,
    route_points: Sequence[Mapping[str, float]],
    progress_m: float,
    lateral_offset_m: float,
    speed_mps: float,
    search_window_m: float,
    max_route_distance_m: float,
    allow_route_fallback: bool = True,
    candidate_offset: int = 0,
) -> Tuple[Any, Dict[str, Any]]:
    blueprints = world.get_blueprint_library()
    walker_blueprints = blueprints.filter("walker.pedestrian.*")
    if not walker_blueprints:
        raise RuntimeError("No pedestrian blueprint found in CARLA.")
    spawn_candidates = _crosswalk_walker_spawn_candidates(
        carla=carla,
        carla_map=carla_map,
        route_points=route_points,
        preferred_progress_m=float(progress_m),
        lateral_offset_m=float(lateral_offset_m),
        search_window_m=float(search_window_m),
        max_route_distance_m=float(max_route_distance_m),
        allow_route_fallback=bool(allow_route_fallback),
    )
    actor = None
    selected_metadata: Dict[str, Any] = {}
    if spawn_candidates:
        offset = int(candidate_offset) % len(spawn_candidates)
        spawn_candidates = spawn_candidates[offset:] + spawn_candidates[:offset]
    pedestrian_candidates = [walker_blueprints[idx] for idx in range(min(len(walker_blueprints), 16))]
    for transform, direction, metadata in spawn_candidates:
        for blueprint in pedestrian_candidates:
            candidate = world.try_spawn_actor(blueprint, transform)
            if candidate is not None:
                actor = candidate
                selected_metadata = {**metadata, "direction_x": direction[0], "direction_y": direction[1]}
                break
        if actor is not None:
            break
    if actor is None:
        raise RuntimeError("Unable to spawn crossing pedestrian on a CARLA crosswalk near the selected route.")
    metadata = {
        "actor_type": str(getattr(actor, "type_id", "")),
        "behavior": "constant_crossing",
        "target_speed_mps": float(speed_mps),
        **selected_metadata,
    }
    return actor, metadata


def _crossing_walker_spec(
    *,
    actor_index: int,
    metadata: Mapping[str, Any],
    progress_m: float,
    lateral_offset_m: float,
    order: int,
    speed_scale: float,
    start_delay_s: float,
) -> Dict[str, Any]:
    target_speed = float(metadata.get("target_speed_mps") or 1.4) * max(float(speed_scale), 0.2)
    return {
        "actor_index": int(actor_index),
        "role": "crossing_pedestrian",
        "kind": "walker",
        "event": "pedestrian_crossing",
        "target_speed_mps": target_speed,
        "trigger_progress_m": float(metadata.get("trigger_progress_m") or 0.0),
        "event_end_progress_m": float(metadata.get("event_end_progress_m") or 0.0),
        "placement": str(metadata.get("placement") or ""),
        "progress_m": float(metadata.get("progress_m") or progress_m),
        "route_lateral_m": float(metadata.get("route_lateral_m") or lateral_offset_m),
        "direction_x": float(metadata.get("direction_x") or 0.0),
        "direction_y": float(metadata.get("direction_y") or 0.0),
        "start_x": float(metadata.get("start_x") if metadata.get("start_x") is not None else metadata.get("x") or 0.0),
        "start_y": float(metadata.get("start_y") if metadata.get("start_y") is not None else metadata.get("y") or 0.0),
        "crosswalk_target_distance_m": float(metadata.get("crosswalk_target_distance_m") or 8.0),
        "start_delay_s": float(start_delay_s),
        "group_order": int(order),
    }


def _crossing_walker_spec_from_runtime_metadata(*, actor_index: int, metadata: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "actor_index": int(actor_index),
        "role": "crossing_pedestrian",
        "kind": "walker",
        "event": "pedestrian_crossing",
        "target_speed_mps": float(metadata.get("target_speed_mps") or 1.4),
        "trigger_progress_m": float(metadata.get("trigger_progress_m") or 0.0),
        "event_end_progress_m": float(metadata.get("event_end_progress_m") or 0.0),
        "placement": str(metadata.get("placement") or "crosswalk"),
        "progress_m": float(metadata.get("progress_m") or 0.0),
        "route_lateral_m": float(metadata.get("route_lateral_m") or 0.0),
        "direction_x": float(metadata.get("direction_x") or 0.0),
        "direction_y": float(metadata.get("direction_y") or 0.0),
        "start_x": float(metadata.get("start_x") if metadata.get("start_x") is not None else metadata.get("x") or 0.0),
        "start_y": float(metadata.get("start_y") if metadata.get("start_y") is not None else metadata.get("y") or 0.0),
        "crosswalk_target_distance_m": float(metadata.get("crosswalk_target_distance_m") or 8.0),
        "start_delay_s": 0.0,
        "group_order": 0,
    }


def _spawn_crossing_walker_group_members(
    *,
    carla: Any,
    world: Any,
    base_metadata: Mapping[str, Any],
    start_index: int,
    count: int,
    group_spacing_m: float,
    group_depth_m: float,
) -> List[Tuple[Any, Dict[str, Any]]]:
    if int(count) <= 0:
        return []
    blueprints = world.get_blueprint_library()
    walker_blueprints = list(blueprints.filter("walker.pedestrian.*"))
    if not walker_blueprints:
        return []
    direction_x = float(base_metadata.get("direction_x") or 0.0)
    direction_y = float(base_metadata.get("direction_y") or 0.0)
    direction_norm = math.hypot(direction_x, direction_y)
    if direction_norm < 1e-6:
        return []
    direction_x /= direction_norm
    direction_y /= direction_norm
    side_x = -direction_y
    side_y = direction_x
    base_x = float(base_metadata.get("start_x") if base_metadata.get("start_x") is not None else base_metadata.get("x") or 0.0)
    base_y = float(base_metadata.get("start_y") if base_metadata.get("start_y") is not None else base_metadata.get("y") or 0.0)
    base_z = float(base_metadata.get("z") or 0.0) + 0.35
    spawned: List[Tuple[Any, Dict[str, Any]]] = []
    for local_idx in range(int(count)):
        order = int(start_index) + local_idx
        side_offset = ((local_idx % 3) - 1) * float(group_spacing_m)
        depth_offset = -float((local_idx // 3) + 1) * float(group_depth_m)
        x = base_x + side_x * side_offset + direction_x * depth_offset
        y = base_y + side_y * side_offset + direction_y * depth_offset
        transform = carla.Transform(
            carla.Location(x=x, y=y, z=base_z),
            carla.Rotation(yaw=math.degrees(math.atan2(direction_y, direction_x))),
        )
        actor = None
        for bp_offset in range(min(len(walker_blueprints), 16)):
            blueprint = walker_blueprints[(order + bp_offset) % len(walker_blueprints)]
            actor = world.try_spawn_actor(blueprint, transform)
            if actor is not None:
                break
        if actor is None:
            continue
        spawned.append(
            (
                actor,
                {
                    "actor_type": str(getattr(actor, "type_id", "")),
                    "group_order": int(order),
                    "x": float(x),
                    "y": float(y),
                    "start_x": float(x),
                    "start_y": float(y),
                    "z": float(base_z),
                },
            )
        )
    return spawned


def _group_crosswalk_locations(locations: Sequence[Any], *, closure_distance_m: float = 0.35) -> List[List[Any]]:
    groups: List[List[Any]] = []
    current: List[Any] = []
    for location in locations:
        if not current:
            current = [location]
            continue
        current.append(location)
        start = current[0]
        distance = math.hypot(float(location.x) - float(start.x), float(location.y) - float(start.y))
        if len(current) >= 4 and distance <= float(closure_distance_m):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _crosswalk_group_axis(group: Sequence[Any]) -> Tuple[float, float]:
    if len(group) < 2:
        return (0.0, 1.0)
    best_pair = (group[0], group[1])
    best_distance = -1.0
    for lhs in group:
        for rhs in group:
            distance = math.hypot(float(rhs.x) - float(lhs.x), float(rhs.y) - float(lhs.y))
            if distance > best_distance:
                best_distance = distance
                best_pair = (lhs, rhs)
    dx = float(best_pair[1].x) - float(best_pair[0].x)
    dy = float(best_pair[1].y) - float(best_pair[0].y)
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return (0.0, 1.0)
    return dx / norm, dy / norm


def _crosswalk_direction_sign(
    *,
    axis: Tuple[float, float],
    route_lateral_axis: Tuple[float, float],
    lateral: float,
    requested_lateral_offset_m: float,
) -> float:
    requested_sign = -1.0 if float(requested_lateral_offset_m) < 0.0 else 1.0
    if abs(float(lateral)) >= 0.5:
        requested_sign = -1.0 if float(lateral) >= 0.0 else 1.0
    dot = float(axis[0]) * float(route_lateral_axis[0]) + float(axis[1]) * float(route_lateral_axis[1])
    if abs(dot) < 1e-6:
        return 1.0
    return requested_sign if dot > 0.0 else -requested_sign


def _crosswalk_target_distance(
    group: Sequence[Any],
    *,
    x: float,
    y: float,
    direction: Tuple[float, float],
) -> float:
    projections = [
        (float(location.x) - float(x)) * float(direction[0]) + (float(location.y) - float(y)) * float(direction[1])
        for location in group
    ]
    min_projection = min(projections or [0.0])
    max_projection = max(projections or [0.0])
    forward_extent = max_projection
    crosswalk_span = max_projection - min_projection
    return max(float(forward_extent) + 1.6, float(crosswalk_span) + 1.8, 7.5)


def _crosswalk_walker_spawn_candidates(
    *,
    carla: Any,
    carla_map: Any,
    route_points: Sequence[Mapping[str, float]],
    preferred_progress_m: float,
    lateral_offset_m: float,
    search_window_m: float,
    max_route_distance_m: float,
    allow_route_fallback: bool = True,
) -> List[Tuple[Any, Tuple[float, float], Dict[str, Any]]]:
    candidates: List[Tuple[float, Any, Tuple[float, float], Dict[str, Any]]] = []
    crosswalk_locations = []
    try:
        crosswalk_locations = list(carla_map.get_crosswalks() or [])
    except Exception:
        crosswalk_locations = []
    crosswalk_groups = _group_crosswalk_locations(crosswalk_locations)
    route_length = _route_length(route_points)
    for group_idx, group in enumerate(crosswalk_groups):
        if not group:
            continue
        projected = []
        for location in group:
            x = float(location.x)
            y = float(location.y)
            progress, lateral, segment_idx = project_to_route((x, y), route_points)
            projected.append((progress, lateral, segment_idx, location))
        valid = [
            item
            for item in projected
            if 0.0 <= float(item[0]) <= route_length
            and abs(float(item[0]) - float(preferred_progress_m)) <= max(float(search_window_m), 1.0)
            and abs(float(item[1])) <= max(float(max_route_distance_m), 1.0)
        ]
        if not valid:
            continue
        preferred_sign = -1.0 if float(lateral_offset_m) < 0.0 else 1.0
        same_side = [item for item in valid if math.copysign(1.0, float(item[1]) or preferred_sign) == preferred_sign]
        side_pool = same_side or valid
        side_pool.sort(
            key=lambda item: (
                abs(abs(float(item[1])) - min(max(abs(float(lateral_offset_m)), 2.0), 6.0)),
                abs(float(item[0]) - float(preferred_progress_m)),
            )
        )
        centroid_x = mean([float(item.x) for item in group])
        centroid_y = mean([float(item.y) for item in group])
        centroid_z = mean([float(getattr(item, "z", 0.0)) for item in group])
        crosswalk_axis = _crosswalk_group_axis(group)
        seen_xy: set[Tuple[int, int]] = set()
        forward_offsets = [0.0, -2.4, 2.4, -4.2, 4.2, -1.2, 1.2]
        for candidate_rank, (base_progress, base_lateral, base_segment_idx, location) in enumerate(side_pool[:8]):
            del base_progress, base_lateral, base_segment_idx
            seed_x = 0.78 * float(location.x) + 0.22 * centroid_x
            seed_y = 0.78 * float(location.y) + 0.22 * centroid_y
            seed_z = 0.78 * float(getattr(location, "z", 0.0)) + 0.22 * centroid_z
            seed_progress, _, _ = project_to_route((seed_x, seed_y), route_points)
            if seed_progress < 0.0 or seed_progress > route_length:
                continue
            seed_pose = _route_pose_at_progress(route_points, seed_progress)
            seed_yaw = float(seed_pose["yaw"])
            forward_axis = (math.cos(seed_yaw), math.sin(seed_yaw))
            for _, forward_offset_m in enumerate(forward_offsets):
                entry_x = seed_x + forward_axis[0] * float(forward_offset_m)
                entry_y = seed_y + forward_axis[1] * float(forward_offset_m)
                z = seed_z
                progress, lateral, segment_idx = project_to_route((entry_x, entry_y), route_points)
                if progress < 0.0 or progress > route_length:
                    continue
                progress_error = abs(progress - float(preferred_progress_m))
                lateral_abs = abs(lateral)
                if progress_error > max(float(search_window_m), 1.0):
                    continue
                if lateral_abs > max(float(max_route_distance_m), 1.0):
                    continue
                route_pose = _route_pose_at_progress(route_points, progress)
                yaw = float(route_pose["yaw"])
                lateral_axis = (-math.sin(yaw), math.cos(yaw))
                direction_sign = _crosswalk_direction_sign(
                    axis=crosswalk_axis,
                    route_lateral_axis=lateral_axis,
                    lateral=float(lateral),
                    requested_lateral_offset_m=float(lateral_offset_m),
                )
                direction = (direction_sign * crosswalk_axis[0], direction_sign * crosswalk_axis[1])
                spawn_setback_m = 2.2
                x = entry_x - direction[0] * spawn_setback_m
                y = entry_y - direction[1] * spawn_setback_m
                xy_key = (int(round(x * 10.0)), int(round(y * 10.0)))
                if xy_key in seen_xy:
                    continue
                seen_xy.add(xy_key)
                spawn_progress, spawn_lateral, _ = project_to_route((x, y), route_points)
                if abs(float(spawn_lateral)) > max(float(max_route_distance_m), 1.0) + 3.0:
                    continue
                transform = carla.Transform(
                    carla.Location(x=x, y=y, z=float(z) + 0.35),
                    carla.Rotation(yaw=math.degrees(math.atan2(direction[1], direction[0]))),
                )
                lateral_penalty = abs(lateral_abs - min(max(abs(float(lateral_offset_m)), 2.0), 6.0))
                score = progress_error + 0.5 * lateral_penalty + 0.04 * abs(float(forward_offset_m)) + 0.01 * candidate_rank
                metadata = {
                    "placement": "crosswalk",
                    "crosswalk_group_index": int(group_idx),
                    "crosswalk_vertex_count": len(crosswalk_locations),
                    "crosswalk_group_count": len(crosswalk_groups),
                    "crosswalk_group_size": len(group),
                    "candidate_rank": int(candidate_rank),
                    "forward_offset_m": float(forward_offset_m),
                    "progress_m": float(progress),
                    "requested_progress_m": float(preferred_progress_m),
                    "route_lateral_m": float(spawn_lateral),
                    "segment_idx": int(segment_idx),
                    "entry_x": float(entry_x),
                    "entry_y": float(entry_y),
                    "spawn_setback_m": float(spawn_setback_m),
                    "spawn_progress_m": float(spawn_progress),
                    "x": x,
                    "y": y,
                    "z": float(z) + 0.35,
                    "start_x": x,
                    "start_y": y,
                    "crosswalk_axis_x": float(crosswalk_axis[0]),
                    "crosswalk_axis_y": float(crosswalk_axis[1]),
                    "crosswalk_target_distance_m": _crosswalk_target_distance(group, x=x, y=y, direction=direction),
                }
                candidates.append((score, transform, direction, metadata))
    candidates.sort(key=lambda item: item[0])
    if candidates:
        return [(transform, direction, metadata) for _, transform, direction, metadata in candidates[:12]]

    if not allow_route_fallback:
        return []

    route_pose = _route_pose_at_progress(route_points, preferred_progress_m)
    yaw = float(route_pose["yaw"])
    lateral_axis = (-math.sin(yaw), math.cos(yaw))
    direction_sign = 1.0 if float(lateral_offset_m) <= 0.0 else -1.0
    transform = _route_transform_at_progress(
        carla=carla,
        route_points=route_points,
        progress_m=float(preferred_progress_m),
        lateral_offset_m=float(lateral_offset_m),
        z_offset_m=0.30,
    )
    metadata = {
        "placement": "route_fallback",
        "crosswalk_vertex_count": len(crosswalk_locations),
        "crosswalk_group_count": len(crosswalk_groups),
        "progress_m": float(preferred_progress_m),
        "requested_progress_m": float(preferred_progress_m),
        "route_lateral_m": float(lateral_offset_m),
        "x": float(transform.location.x),
        "y": float(transform.location.y),
        "z": float(transform.location.z),
    }
    return [(transform, (direction_sign * lateral_axis[0], direction_sign * lateral_axis[1]), metadata)]


def _route_transform_at_progress(
    *,
    carla: Any,
    route_points: Sequence[Mapping[str, float]],
    progress_m: float,
    lateral_offset_m: float,
    z_offset_m: float,
) -> Any:
    pose = _route_pose_at_progress(route_points, progress_m)
    yaw = float(pose["yaw"])
    x = float(pose["x"]) - math.sin(yaw) * float(lateral_offset_m)
    y = float(pose["y"]) + math.cos(yaw) * float(lateral_offset_m)
    z = float(pose.get("z") or 0.0) + float(z_offset_m)
    return carla.Transform(
        carla.Location(x=x, y=y, z=z),
        carla.Rotation(yaw=math.degrees(yaw)),
    )


def _route_pose_at_progress(route: Sequence[Mapping[str, float]], progress_m: float) -> Dict[str, float]:
    sampled = dict(_sample_route_at_progress(route, progress_m))
    yaw = sampled.get("yaw")
    if yaw is None:
        ahead = _sample_route_at_progress(route, float(progress_m) + 2.0)
        yaw = math.atan2(float(ahead["y"]) - float(sampled["y"]), float(ahead["x"]) - float(sampled["x"]))
    sampled["yaw"] = float(yaw)
    return sampled


def _update_scenario_runtime(
    *,
    carla: Any,
    runtime: CarlaScenarioRuntime,
    step_idx: int,
    traffic_manager: Optional[Any] = None,
    carla_map: Optional[Any] = None,
    ego: Optional[Any] = None,
    route_points: Optional[Sequence[Mapping[str, float]]] = None,
) -> None:
    if runtime.scenario_type in {"dense_follow_overtake", "adjacent_lane_cut_in"} and runtime.actors:
        _update_scenario_event_state(
            carla=carla,
            runtime=runtime,
            ego=ego,
            route_points=route_points or [],
        )
    elif runtime.scenario_type == "pedestrian_crossing" and runtime.actors:
        _update_pedestrian_crossing_runtime(
            carla=carla,
            runtime=runtime,
            ego=ego,
            route_points=route_points or [],
        )


def _update_pedestrian_crossing_runtime(
    *,
    carla: Any,
    runtime: CarlaScenarioRuntime,
    ego: Optional[Any],
    route_points: Sequence[Mapping[str, float]],
) -> None:
    ego_progress = 0.0
    if ego is not None and route_points:
        transform = ego.get_transform()
        ego_progress, _, _ = project_to_route((float(transform.location.x), float(transform.location.y)), route_points)
    update_tick = int(runtime.metadata.get("_update_tick") or 0)
    runtime.metadata["_update_tick"] = update_tick + 1
    fps = max(int(runtime.metadata.get("fps") or 10), 1)
    trigger = float(runtime.metadata.get("trigger_progress_m") or 0.0)
    event_end = float(runtime.metadata.get("event_end_progress_m") or trigger + 24.0)
    actor_specs = [dict(spec) for spec in list(runtime.metadata.get("actor_specs") or [])]
    if not actor_specs and runtime.actors:
        actor_specs = [_crossing_walker_spec_from_runtime_metadata(actor_index=0, metadata=runtime.metadata)]
    walker_specs = [spec for spec in actor_specs if str(spec.get("kind") or "") == "walker"]
    if not walker_specs:
        runtime.metadata["active_event"] = "route_following"
        runtime.metadata["active_events"] = ["route_following"]
        return

    trigger_mode = str(runtime.metadata.get("pedestrian_trigger_mode") or "ego_progress").strip().lower()
    if trigger_mode in {"time", "time_based", "rollout_time"}:
        trigger_tick = max(int(round(max(float(runtime.metadata.get("pedestrian_trigger_time_s") or 0.0), 0.0) * fps)), 0)
        group_triggered = bool(runtime.metadata.get("pedestrian_group_triggered")) or update_tick >= trigger_tick
    else:
        trigger_buffer_m = max(float(runtime.metadata.get("pedestrian_trigger_buffer_m") or 0.0), 0.0)
        group_triggered = bool(runtime.metadata.get("pedestrian_group_triggered")) or ego_progress >= max(
            trigger - trigger_buffer_m,
            0.0,
        )
        trigger_tick = update_tick
    if group_triggered and not bool(runtime.metadata.get("pedestrian_group_triggered")):
        runtime.metadata["pedestrian_group_trigger_tick"] = trigger_tick
    runtime.metadata["pedestrian_group_triggered"] = bool(group_triggered)
    raw_trigger_tick = runtime.metadata.get("pedestrian_group_trigger_tick")
    trigger_tick = int(raw_trigger_tick) if raw_trigger_tick is not None else update_tick

    any_started = False
    completions: List[float] = []
    all_complete = True
    for spec in actor_specs:
        actor_index = int(spec.get("actor_index") or 0)
        if actor_index < 0 or actor_index >= len(runtime.actors):
            continue
        if str(spec.get("kind") or "") != "walker":
            continue
        actor = runtime.actors[actor_index]
        target_distance = float(spec.get("crosswalk_target_distance_m") or 8.0)
        crossing_progress = _walker_crossing_progress(actor, spec)
        completion = max(min(float(crossing_progress) / max(target_distance, 1e-6), 1.0), 0.0)
        crossing_complete = completion >= 0.995
        completions.append(completion)
        all_complete = all_complete and crossing_complete
        delay_steps = max(int(round(max(float(spec.get("start_delay_s") or 0.0), 0.0) * fps)), 0)
        started_key = f"pedestrian_started_{actor_index}"
        pedestrian_started = (
            bool(runtime.metadata.get(started_key))
            or (group_triggered and update_tick >= trigger_tick + delay_steps)
        )
        runtime.metadata[started_key] = bool(pedestrian_started)
        any_started = any_started or pedestrian_started
        should_move = pedestrian_started and not crossing_complete
        direction = carla.Vector3D(
            x=float(spec.get("direction_x") or 0.0),
            y=float(spec.get("direction_y") or 0.0),
            z=0.0,
        )
        actor.apply_control(
            carla.WalkerControl(
                direction=direction,
                speed=float(spec.get("target_speed_mps") or 1.2) if should_move else 0.0,
            )
        )

    runtime.metadata["ego_progress_m"] = float(ego_progress)
    runtime.metadata["pedestrian_started"] = bool(any_started)
    runtime.metadata["pedestrian_completed"] = bool(all_complete)
    runtime.metadata["walker_group_min_completion"] = min(completions) if completions else 0.0
    runtime.metadata["walker_group_max_completion"] = max(completions) if completions else 0.0
    if all_complete:
        active_event = "pedestrian_cleared"
    elif any_started or trigger <= ego_progress <= event_end:
        active_event = "pedestrian_crossing"
    else:
        active_event = "route_following"
    runtime.metadata["active_event"] = active_event
    runtime.metadata["active_events"] = [active_event]


def _update_scenario_event_state(
    *,
    carla: Any,
    runtime: CarlaScenarioRuntime,
    ego: Optional[Any],
    route_points: Sequence[Mapping[str, float]],
) -> None:
    ego_progress = 0.0
    if ego is not None and route_points:
        transform = ego.get_transform()
        ego_progress, _, _ = project_to_route((float(transform.location.x), float(transform.location.y)), route_points)

    active_events: List[str] = []
    for passive in list(runtime.metadata.get("passive_events") or []):
        start = float(dict(passive).get("event_start_progress_m") or -1.0)
        end = float(dict(passive).get("event_end_progress_m") or -1.0)
        event = str(dict(passive).get("event") or "")
        if event and start <= ego_progress <= end:
            active_events.append(event)
    actor_specs = list(runtime.metadata.get("actor_specs") or [])
    for spec in actor_specs:
        actor_index = int(spec.get("actor_index") or 0)
        if actor_index < 0 or actor_index >= len(runtime.actors):
            continue
        actor = runtime.actors[actor_index]
        role = str(spec.get("role") or "")
        kind = str(spec.get("kind") or "vehicle")
        event = str(spec.get("event") or role)
        if kind == "walker":
            trigger = float(spec.get("trigger_progress_m") or 0.0)
            end = float(spec.get("event_end_progress_m") or trigger + 24.0)
            crossing_progress = _walker_crossing_progress(actor, spec)
            target_distance = float(spec.get("crosswalk_target_distance_m") or 8.0)
            crossing_complete = crossing_progress >= target_distance
            started_key = f"pedestrian_started_{actor_index}"
            pedestrian_started = bool(spec.get("pedestrian_started")) or bool(runtime.metadata.get(started_key)) or ego_progress >= trigger
            spec["pedestrian_started"] = bool(pedestrian_started)
            runtime.metadata[started_key] = bool(pedestrian_started)
            should_move = pedestrian_started and not crossing_complete
            if crossing_complete:
                active_events.append("pedestrian_cleared")
            elif pedestrian_started or trigger <= ego_progress <= end:
                active_events.append(event)
            direction = carla.Vector3D(
                x=float(spec.get("direction_x") or 0.0),
                y=float(spec.get("direction_y") or 0.0),
                z=0.0,
            )
            actor.apply_control(
                carla.WalkerControl(
                    direction=direction,
                    speed=float(spec.get("target_speed_mps") or 1.2) if should_move else 0.0,
                )
            )
            continue

        if kind == "autopilot_vehicle":
            start = float(spec.get("event_start_progress_m") or -1.0)
            end = float(spec.get("event_end_progress_m") or -1.0)
            if start <= ego_progress <= end:
                active_events.append(event)
            continue

        start = float(spec.get("event_start_progress_m") or -1.0)
        end = float(spec.get("event_end_progress_m") or -1.0)
        if start <= ego_progress <= end:
            active_events.append(event)

    unique_events = []
    for event in active_events:
        if event and event not in unique_events:
            unique_events.append(event)
    runtime.metadata["ego_progress_m"] = float(ego_progress)
    runtime.metadata["active_events"] = unique_events or ["route_following"]
    runtime.metadata["active_event"] = ", ".join(runtime.metadata["active_events"])


def _attach_rgb_cameras(
    *,
    carla: Any,
    world: Any,
    ego: Any,
    config: CarlaVisionClosedLoopConfig,
) -> Tuple[List[Any], Dict[str, "queue.Queue[Any]"]]:
    camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(int(config.camera_width)))
    camera_bp.set_attribute("image_size_y", str(int(config.camera_height)))
    camera_bp.set_attribute("fov", str(max(float(config.camera_fov), 100.0)))
    if camera_bp.has_attribute("enable_postprocess_effects"):
        camera_bp.set_attribute("enable_postprocess_effects", "true")
    sensors = []
    queues: Dict[str, "queue.Queue[Any]"] = {}
    for camera_name in DEFAULT_BENCH2DRIVE_CAMERAS:
        spec = CAMERA_TRANSFORMS[camera_name]
        transform = carla.Transform(
            carla.Location(x=spec["x"], y=spec["y"], z=spec["z"]),
            carla.Rotation(pitch=spec["pitch"], yaw=spec["yaw"], roll=spec["roll"]),
        )
        sensor = world.spawn_actor(camera_bp, transform, attach_to=ego)
        image_queue: "queue.Queue[Any]" = queue.Queue()
        sensor.listen(image_queue.put)
        sensors.append(sensor)
        queues[camera_name] = image_queue
    return sensors, queues


def _attach_video_camera(
    *,
    carla: Any,
    world: Any,
    ego: Any,
    config: CarlaVisionClosedLoopConfig,
) -> Tuple[Optional[Any], Optional["queue.Queue[Any]"]]:
    width = int(config.video_width or 0)
    height = int(config.video_height or 0)
    if width <= 0 or height <= 0:
        return None, None
    camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(width))
    camera_bp.set_attribute("image_size_y", str(height))
    camera_bp.set_attribute("fov", str(float(config.camera_fov)))
    if camera_bp.has_attribute("enable_postprocess_effects"):
        camera_bp.set_attribute("enable_postprocess_effects", "true")
    spec = CAMERA_TRANSFORMS["rgb_front"]
    transform = carla.Transform(
        carla.Location(x=-6.5, y=spec["y"], z=3.2),
        carla.Rotation(pitch=-12.0, yaw=spec["yaw"], roll=spec["roll"]),
    )
    sensor = world.spawn_actor(camera_bp, transform, attach_to=ego)
    image_queue: "queue.Queue[Any]" = queue.Queue()
    sensor.listen(image_queue.put)
    return sensor, image_queue


def _attach_collision_sensor(*, carla: Any, world: Any, ego: Any, events: List[Dict[str, Any]]) -> Any:
    collision_bp = world.get_blueprint_library().find("sensor.other.collision")
    sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego)

    def _callback(event: Any) -> None:
        impulse = event.normal_impulse
        events.append(
            {
                "frame": int(event.frame),
                "actor_type": str(getattr(event.other_actor, "type_id", "")),
                "impulse": float(math.sqrt(impulse.x * impulse.x + impulse.y * impulse.y + impulse.z * impulse.z)),
            }
        )

    sensor.listen(_callback)
    return sensor


def _drain_camera_queues(camera_queues: Mapping[str, "queue.Queue[Any]"]) -> None:
    for image_queue in camera_queues.values():
        _drain_queue(image_queue)


def _drain_queue(image_queue: "queue.Queue[Any]") -> None:
    while True:
        try:
            image_queue.get_nowait()
        except queue.Empty:
            break


def _nearest_scenario_actor_distance(*, ego: Any, runtime: CarlaScenarioRuntime) -> float:
    if not runtime.actors:
        return -1.0
    ego_location = ego.get_location()
    distances = []
    for actor in runtime.actors:
        try:
            distances.append(float(ego_location.distance(actor.get_location())))
        except Exception:
            continue
    return min(distances) if distances else -1.0


def _mean_scenario_actor_speed(runtime: CarlaScenarioRuntime) -> float:
    speeds = []
    for actor in runtime.actors:
        try:
            speeds.append(_actor_speed_mps(actor))
        except Exception:
            continue
    return mean(speeds) if speeds else 0.0


def _natural_traffic_frame_context(
    *,
    ego: Any,
    runtime: CarlaScenarioRuntime,
    route_points: Sequence[Mapping[str, float]],
) -> Dict[str, float]:
    specs = {
        int(dict(spec).get("actor_index") or 0): dict(spec)
        for spec in list(runtime.metadata.get("actor_specs") or [])
    }
    ego_transform = ego.get_transform()
    ego_location = ego_transform.location
    ego_progress = 0.0
    ego_lateral = 0.0
    if route_points:
        ego_progress, ego_lateral, _ = project_to_route(
            (float(ego_location.x), float(ego_location.y)),
            route_points,
        )

    distances: List[float] = []
    speeds: List[float] = []
    visible_count = 0
    nearby_count = 0
    front_count = 0
    adjacent_count = 0
    same_lane_front_count = 0
    for idx, actor in enumerate(runtime.actors):
        spec = specs.get(idx, {})
        if str(spec.get("role") or "") != "ambient_autopilot_vehicle":
            continue
        try:
            location = actor.get_location()
        except Exception:
            continue
        distance = float(ego_location.distance(location))
        distances.append(distance)
        try:
            speeds.append(_actor_speed_mps(actor))
        except Exception:
            pass
        forward_m, right_m = _world_to_ego_xy(ego_transform, float(location.x), float(location.y))
        route_gap = forward_m
        route_lateral_gap = right_m
        if route_points:
            actor_progress, actor_lateral, _ = project_to_route(
                (float(location.x), float(location.y)),
                route_points,
            )
            route_gap = float(actor_progress) - float(ego_progress)
            route_lateral_gap = float(actor_lateral) - float(ego_lateral)
        if 0.0 <= forward_m <= 70.0 and abs(float(right_m)) <= 30.0:
            visible_count += 1
        if distance <= 30.0:
            nearby_count += 1
        if 0.0 <= route_gap <= 40.0 and abs(float(route_lateral_gap)) <= 8.0:
            front_count += 1
        if 0.0 <= route_gap <= 40.0 and 2.0 < abs(float(route_lateral_gap)) <= 12.0:
            adjacent_count += 1
        if 0.0 <= route_gap <= 35.0 and abs(float(route_lateral_gap)) <= 2.5:
            same_lane_front_count += 1
    return {
        "natural_traffic_actor_count": float(len(distances)),
        "natural_traffic_nearest_distance_m": min(distances) if distances else -1.0,
        "natural_traffic_mean_speed_mps": mean(speeds) if speeds else 0.0,
        "natural_traffic_visible_actor_count": float(visible_count),
        "natural_traffic_nearby_actor_count": float(nearby_count),
        "natural_traffic_front_actor_count": float(front_count),
        "natural_traffic_adjacent_actor_count": float(adjacent_count),
        "natural_traffic_same_lane_front_actor_count": float(same_lane_front_count),
    }


def _pedestrian_crossing_frame_context(
    *,
    runtime: CarlaScenarioRuntime,
    route_points: Sequence[Mapping[str, float]],
) -> Dict[str, float]:
    specs = {
        int(dict(spec).get("actor_index") or 0): dict(spec)
        for spec in list(runtime.metadata.get("actor_specs") or [])
    }
    progress_values: List[float] = []
    lateral_values: List[float] = []
    crossing_values: List[float] = []
    target_values: List[float] = []
    completion_values: List[float] = []
    for idx, actor in enumerate(runtime.actors):
        spec = specs.get(idx, {})
        if str(spec.get("kind") or "") != "walker":
            continue
        try:
            location = actor.get_location()
        except Exception:
            continue
        progress, lateral, _ = project_to_route((float(location.x), float(location.y)), route_points)
        target_distance = float(spec.get("crosswalk_target_distance_m") or 8.0)
        crossing_progress = _walker_crossing_progress(actor, spec)
        completion = max(min(float(crossing_progress) / max(target_distance, 1e-6), 1.0), 0.0)
        progress_values.append(float(progress))
        lateral_values.append(float(lateral))
        crossing_values.append(float(crossing_progress))
        target_values.append(float(target_distance))
        completion_values.append(float(completion))
    if completion_values:
        least_complete_idx = min(range(len(completion_values)), key=lambda idx: completion_values[idx])
        return {
            "walker_route_progress_m": float(progress_values[least_complete_idx]),
            "walker_route_lateral_m": float(lateral_values[least_complete_idx]),
            "walker_crossing_progress_m": float(min(crossing_values)),
            "walker_crossing_target_distance_m": float(max(target_values)),
            "walker_crossing_completion": float(min(completion_values)),
        }
    return {
        "walker_route_progress_m": -1.0,
        "walker_route_lateral_m": 0.0,
        "walker_crossing_progress_m": 0.0,
        "walker_crossing_target_distance_m": 0.0,
        "walker_crossing_completion": 0.0,
    }


def _driving_lane_frame_context(
    *,
    carla: Any,
    carla_map: Any,
    x: float,
    y: float,
    z: float,
) -> Dict[str, float]:
    try:
        location = carla.Location(x=float(x), y=float(y), z=float(z))
        waypoint = carla_map.get_waypoint(location, project_to_road=False, lane_type=carla.LaneType.Driving)
        projected = carla_map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
    except Exception:
        return {
            "ego_on_driving_lane": 1.0,
            "ego_nearest_driving_lane_center_distance_m": 0.0,
            "ego_driving_lane_width_m": 0.0,
        }
    nearest_distance = -1.0
    lane_width = 0.0
    if projected is not None:
        try:
            nearest_distance = float(location.distance(projected.transform.location))
            lane_width = float(getattr(projected, "lane_width", 0.0) or 0.0)
        except Exception:
            nearest_distance = -1.0
            lane_width = 0.0
    distance_limit = max(float(lane_width) * 0.60, 2.05) if lane_width > 0.0 else 2.05
    projected_on_driving_lane = nearest_distance >= 0.0 and nearest_distance <= distance_limit
    return {
        "ego_on_driving_lane": 1.0 if (waypoint is not None or projected_on_driving_lane) else 0.0,
        "ego_nearest_driving_lane_center_distance_m": nearest_distance,
        "ego_driving_lane_width_m": lane_width,
    }


def _walker_crossing_progress(actor: Any, metadata: Mapping[str, Any]) -> float:
    try:
        location = actor.get_location()
    except Exception:
        return 0.0
    start_x = float(metadata.get("start_x") if metadata.get("start_x") is not None else metadata.get("x") or location.x)
    start_y = float(metadata.get("start_y") if metadata.get("start_y") is not None else metadata.get("y") or location.y)
    direction_x = float(metadata.get("direction_x") or 0.0)
    direction_y = float(metadata.get("direction_y") or 0.0)
    norm = math.hypot(direction_x, direction_y)
    if norm < 1e-6:
        return 0.0
    dx = float(location.x) - start_x
    dy = float(location.y) - start_y
    return max((dx * direction_x + dy * direction_y) / norm, 0.0)


def _actor_speed_mps(actor: Any) -> float:
    velocity = actor.get_velocity()
    return float(math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z))


def _rollout_vision_planner(
    *,
    carla: Any,
    world: Any,
    ego: Any,
    model: Any,
    torch: Any,
    device: Any,
    model_config: VisionE2EModelConfig,
    route_trace: Sequence[Any],
    route_points: Sequence[Mapping[str, float]],
    camera_queues: Mapping[str, "queue.Queue[Any]"],
    video_queue: Optional["queue.Queue[Any]"],
    scenario_runtime: CarlaScenarioRuntime,
    traffic_manager: Optional[Any],
    carla_map: Any,
    collision_events: Sequence[Mapping[str, Any]],
    config: CarlaVisionClosedLoopConfig,
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    del route_trace, model_config
    states: List[Dict[str, Any]] = []
    frames = []
    model_waypoint_controller = PurePursuitConfig(lookahead_m=MODEL_WAYPOINT_LOOKAHEAD_M, speed_kp=0.35)
    max_steps = max(int(float(config.horizon_s) * int(config.fps)), 1)
    previous_yaw = None
    previous_accel = 0.0
    previous_speed = 0.0
    with torch.no_grad():
        for step_idx in range(max_steps):
            _update_scenario_runtime(
                carla=carla,
                runtime=scenario_runtime,
                step_idx=step_idx,
                traffic_manager=traffic_manager,
                carla_map=carla_map,
                ego=ego,
                route_points=route_points,
            )
            traffic_light_details = _condition_ego_route_traffic_lights(
                carla=carla,
                world=world,
                ego=ego,
                route_points=route_points,
                runtime=scenario_runtime,
                enabled=bool(config.condition_ego_route_traffic_lights),
            )
            snapshot = world.tick()
            camera_images = _read_synced_camera_images(camera_queues, frame_id=int(snapshot), timeout_s=5.0)
            video_image = None
            if video_queue is not None:
                video_image = _read_synced_queue_image(video_queue, frame_id=int(snapshot), timeout_s=5.0)
            transform = ego.get_transform()
            velocity = ego.get_velocity()
            speed_mps = math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z)
            route_features, command = _build_carla_route_features(
                transform=transform,
                speed_mps=speed_mps,
                route_points=route_points,
            )
            images_tensor = _camera_images_to_model_tensor(
                torch=torch,
                camera_images=camera_images,
                image_size=int(config.image_size),
                device=device,
            )
            route_tensor = torch.tensor([route_features], dtype=torch.float32, device=device)
            prediction = model(images_tensor, route_tensor)
            pred_waypoints = prediction["future"].detach().cpu().reshape(-1, 2).tolist()
            pred_control = prediction["control"].detach().cpu().reshape(-1).tolist()
            brake_probability = float(torch.sigmoid(prediction["brake_logits"]).detach().cpu().reshape(-1)[0])
            prediction_stats = _prediction_path_stats(pred_waypoints)
            control, control_details = _carla_control_from_prediction(
                state=_carla_state_from_actor(ego),
                transform=transform,
                speed_mps=speed_mps,
                pred_waypoints=pred_waypoints,
                pred_control=pred_control,
                brake_probability=brake_probability,
                config=config,
                controller_config=model_waypoint_controller,
            )
            control, behavior_details = _apply_behavior_override(
                runtime=scenario_runtime,
                control=control,
            )
            if bool(config.enable_scenario_safety_override):
                control, safety_details = _apply_scenario_safety_override(
                    carla=carla,
                    ego=ego,
                    route_points=route_points,
                    runtime=scenario_runtime,
                    control=control,
                )
            else:
                safety_details = {"safety_brake": 0.0}
            lane_guard_details = {"behavior_override": ""}
            if bool(config.enable_lane_departure_guard) and float(safety_details.get("safety_brake") or 0.0) <= 0.05:
                control, lane_guard_details = _apply_lane_departure_guard(
                    carla=carla,
                    carla_map=carla_map,
                    ego=ego,
                    route_points=route_points,
                    control=control,
                )
                if str(lane_guard_details.get("behavior_override") or ""):
                    behavior_details = {
                        **behavior_details,
                        "behavior_override": str(lane_guard_details.get("behavior_override") or ""),
                    }
            lane_guard_active = bool(str(lane_guard_details.get("behavior_override") or ""))
            if not bool(config.enable_lane_departure_guard):
                lane_guard_details = {"behavior_override": ""}
            ego.apply_control(carla.VehicleControl(**control))

            yaw_rad = math.radians(float(transform.rotation.yaw))
            yaw_rate = 0.0 if previous_yaw is None else normalize_angle(yaw_rad - previous_yaw) * int(config.fps)
            previous_yaw = yaw_rad
            acceleration = (speed_mps - previous_speed) * int(config.fps)
            jerk = (acceleration - previous_accel) * int(config.fps)
            previous_speed = speed_mps
            previous_accel = acceleration
            progress, lateral_error, _ = project_to_route(
                (float(transform.location.x), float(transform.location.y)),
                route_points,
            )
            route_completion = progress / max(_route_length(route_points), 1e-6)
            lane_context = _driving_lane_frame_context(
                carla=carla,
                carla_map=carla_map,
                x=float(transform.location.x),
                y=float(transform.location.y),
                z=float(transform.location.z),
            )
            scenario_distance = _nearest_scenario_actor_distance(ego=ego, runtime=scenario_runtime)
            scenario_speed = _mean_scenario_actor_speed(runtime=scenario_runtime)
            natural_traffic_context = _natural_traffic_frame_context(
                ego=ego,
                runtime=scenario_runtime,
                route_points=route_points,
            )
            pedestrian_context = _pedestrian_crossing_frame_context(
                runtime=scenario_runtime,
                route_points=route_points,
            )
            state_row = {
                "step": step_idx,
                "frame": int(snapshot),
                "t_s": step_idx / max(int(config.fps), 1),
                "scenario_name": scenario_runtime.name,
                "scenario_type": scenario_runtime.scenario_type,
                "scenario_phase": str(scenario_runtime.metadata.get("active_event") or ""),
                "x": float(transform.location.x),
                "y": float(transform.location.y),
                "z": float(transform.location.z),
                "yaw": yaw_rad,
                "speed_mps": speed_mps,
                "acceleration_mps2": acceleration,
                "jerk_mps3": jerk,
                "yaw_rate_rps": yaw_rate,
                "route_progress_m": progress,
                "route_completion": max(min(route_completion, 1.0), 0.0),
                "lateral_error_m": lateral_error,
                "steer": float(control["steer"]),
                "throttle": float(control["throttle"]),
                "brake": float(control["brake"]),
                "target_speed_mps": float(control_details["target_speed_mps"]),
                "navigation_route_offset_m": float(control_details.get("navigation_route_offset_m") or 0.0),
                "behavior_override": str(behavior_details.get("behavior_override") or ""),
                "brake_probability": brake_probability,
                "command": float(command),
                "ego_control_mode": str(control_details["ego_control_mode"]),
                "model_waypoint_controller_weight": float(
                    control_details.get("model_waypoint_controller_weight") or 0.0
                ),
                "network_control_blend_weight": float(control_details.get("network_control_blend_weight") or 0.0),
                "control_source": str(control_details.get("control_source") or ""),
                "pred_waypoint_count": int(prediction_stats["count"]),
                "pred_path_length_m": float(prediction_stats["path_length_m"]),
                "pred_final_forward_m": float(prediction_stats["final_forward_m"]),
                "pred_final_right_m": float(prediction_stats["final_right_m"]),
                "pred_mean_abs_right_m": float(prediction_stats["mean_abs_right_m"]),
                "pred_control_steer": float(pred_control[0]) if len(pred_control) >= 1 else 0.0,
                "pred_control_throttle": float(pred_control[1]) if len(pred_control) >= 2 else 0.0,
                "pred_control_brake": float(pred_control[2]) if len(pred_control) >= 3 else 0.0,
                "pred_waypoints_ego_json": json.dumps(pred_waypoints, separators=(",", ":")),
                "safety_brake": float(safety_details.get("safety_brake") or 0.0),
                "safety_actor_role": str(safety_details.get("actor_role") or ""),
                "safety_gap_m": float(safety_details.get("gap_m") or -1.0),
                "safety_actor_distance_m": float(safety_details.get("distance_m") or -1.0),
                "lane_guard_active": 1.0 if lane_guard_active else 0.0,
                "traffic_light_conditioned": 1.0 if bool(traffic_light_details.get("enabled")) else 0.0,
                "ego_route_traffic_light_id": int(traffic_light_details.get("selected_light_id") or -1),
                "ego_route_traffic_light_group_size": int(traffic_light_details.get("group_size") or 0),
                "collision_count": len(collision_events),
                "scenario_actor_distance_m": scenario_distance,
                "scenario_actor_speed_mps": scenario_speed,
                **lane_context,
                **natural_traffic_context,
                **pedestrian_context,
            }
            states.append(state_row)
            frame_source = video_image if video_image is not None else camera_images["rgb_front"]
            frames.append(_render_front_frame(frame_source, state_row, pred_waypoints=pred_waypoints))
            if len(collision_events) > 0:
                break
            if route_completion >= 0.98 or _pedestrian_yield_rollout_complete(states, fps=int(config.fps)):
                scenario_runtime.metadata["semantic_completion_reason"] = "pedestrian_yield_complete"
                break
    return states, frames


def _pedestrian_yield_rollout_complete(states: Sequence[Mapping[str, Any]], *, fps: int) -> bool:
    if not states:
        return False
    latest = states[-1]
    if str(latest.get("scenario_type") or "") != "pedestrian_crossing":
        return False
    first_complete_idx = next(
        (
            idx
            for idx, row in enumerate(states)
            if float(row.get("walker_crossing_completion") or 0.0) >= 0.95
        ),
        None,
    )
    if first_complete_idx is None:
        return False
    post_rows = list(states[first_complete_idx:])
    if len(post_rows) < max(int(float(fps) * 0.8), 6):
        return False
    if max(float(row.get("speed_mps") or 0.0) for row in post_rows) < 0.75:
        return False
    start_progress = float(states[first_complete_idx].get("route_progress_m") or 0.0)
    max_post_progress = max(float(row.get("route_progress_m") or 0.0) for row in post_rows)
    if max_post_progress - start_progress < 6.0:
        return False
    latest_phase = str(latest.get("scenario_phase") or "")
    if "pedestrian_cleared" not in latest_phase:
        return False
    commands_after_clear = [int(float(row.get("command") or 4.0)) for row in post_rows]
    right_turn_frames = sum(1 for command in commands_after_clear if command == 2)
    saw_right_turn_context = right_turn_frames >= max(int(float(fps) * 1.5), 8)
    returned_to_lane_follow = any(command == 4 for command in commands_after_clear[max(len(commands_after_clear) // 3, 1) :])
    if saw_right_turn_context and returned_to_lane_follow:
        return True
    return False


def _read_synced_camera_images(
    camera_queues: Mapping[str, "queue.Queue[Any]"],
    *,
    frame_id: int,
    timeout_s: float,
) -> Dict[str, Any]:
    images = {}
    for camera_name, image_queue in camera_queues.items():
        image = None
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            try:
                candidate = image_queue.get(timeout=max(min(deadline - time.time(), 0.5), 0.01))
                if int(candidate.frame) >= int(frame_id):
                    image = candidate
                    break
            except queue.Empty:
                continue
        if image is None:
            raise RuntimeError(f"Timed out waiting for CARLA camera {camera_name} at frame {frame_id}.")
        images[str(camera_name)] = image
    return images


def _read_synced_queue_image(image_queue: "queue.Queue[Any]", *, frame_id: int, timeout_s: float) -> Any:
    image = None
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        try:
            candidate = image_queue.get(timeout=max(min(deadline - time.time(), 0.5), 0.01))
            if int(candidate.frame) >= int(frame_id):
                image = candidate
                break
        except queue.Empty:
            continue
    if image is None:
        raise RuntimeError(f"Timed out waiting for CARLA video camera at frame {frame_id}.")
    return image


def _camera_images_to_model_tensor(
    *,
    torch: Any,
    camera_images: Mapping[str, Any],
    image_size: int,
    device: Any,
) -> Any:
    arrays = []
    for camera_name in DEFAULT_BENCH2DRIVE_CAMERAS:
        arrays.append(_carla_image_to_chw_uint8(camera_images[camera_name], image_size=image_size))
    import numpy as np

    stacked = np.stack(arrays, axis=0)
    images = torch.from_numpy(stacked.copy()).unsqueeze(0).to(device, non_blocking=True).float().div_(255.0)
    return _normalize_images_on_device(images)


def _carla_image_to_chw_uint8(image: Any, *, image_size: int) -> Any:
    import numpy as np
    from PIL import Image

    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
    array = array[:, :, ::-1]
    pil = Image.fromarray(array, mode="RGB").resize((int(image_size), int(image_size)))
    return np.transpose(np.asarray(pil, dtype=np.uint8), (2, 0, 1))


def _carla_image_to_rgb_array(image: Any) -> Any:
    import numpy as np

    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
    return array[:, :, ::-1].copy()


def _render_front_frame(
    image: Any,
    state: Mapping[str, Any],
    *,
    pred_waypoints: Optional[Sequence[Sequence[float]]] = None,
) -> Any:
    from PIL import Image, ImageDraw, ImageFont

    pil = Image.fromarray(_carla_image_to_rgb_array(image), mode="RGB")
    draw = ImageDraw.Draw(pil)
    font_size = max(12, int(round(pil.size[1] / 62.0)))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    phase = str(state.get("scenario_phase") or "route_following")
    phase_tokens = [item.strip() for item in phase.split(",") if item.strip()]
    compact_phase = ", ".join(phase_tokens[:2])
    if len(phase_tokens) > 2:
        compact_phase += ", ..."
    lines = [
        f"{float(state['t_s']):.1f}s  {compact_phase}",
        f"speed {float(state['speed_mps']):.1f} m/s  brake {float(state['brake']):.2f}  safety {float(state.get('safety_brake') or 0.0):.2f}",
        f"pred {float(state.get('pred_path_length_m') or 0.0):.1f} m  ctrl [{float(state.get('pred_control_steer') or 0.0):+.2f}, {float(state.get('pred_control_throttle') or 0.0):.2f}, {float(state.get('pred_control_brake') or 0.0):.2f}]",
    ]
    x, y = max(12, pil.size[0] // 120), max(12, pil.size[1] // 80)
    pad_x = max(4, font_size // 4)
    pad_y = max(2, font_size // 6)
    line_gap = max(2, font_size // 5)
    for line in lines:
        bbox = draw.textbbox((x, y), line, font=font)
        draw.rectangle(
            (bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y),
            fill=(0, 0, 0),
        )
        draw.text((x, y), line, fill=(255, 255, 255), font=font)
        y += (bbox[3] - bbox[1]) + 2 * pad_y + line_gap
    _draw_prediction_inset(draw=draw, image_size=pil.size, pred_waypoints=pred_waypoints or [], font=font)
    return pil


def _draw_prediction_inset(
    *,
    draw: Any,
    image_size: Tuple[int, int],
    pred_waypoints: Sequence[Sequence[float]],
    font: Any,
) -> None:
    width, height = image_size
    inset_w = max(int(width * 0.18), 150)
    inset_h = max(int(height * 0.26), 100)
    margin = max(12, width // 100)
    left = width - inset_w - margin
    top = height - inset_h - margin
    right = width - margin
    bottom = height - margin
    draw.rectangle((left, top, right, bottom), fill=(0, 0, 0), outline=(230, 230, 230), width=1)
    cx = left + inset_w // 2
    cy = bottom - max(14, inset_h // 10)
    draw.line((cx, top + 12, cx, bottom - 8), fill=(95, 95, 95), width=1)
    draw.line((left + 8, cy, right - 8, cy), fill=(95, 95, 95), width=1)
    draw.polygon([(cx, cy - 10), (cx - 6, cy + 6), (cx + 6, cy + 6)], fill=(255, 255, 255))
    points: List[Tuple[float, float]] = []
    for waypoint in pred_waypoints:
        if len(waypoint) < 2:
            continue
        forward_m, right_m = _bench2drive_waypoint_to_forward_right(waypoint)
        px = cx + float(right_m) * (inset_w / 14.0)
        py = cy - float(forward_m) * (inset_h / 34.0)
        px = max(min(px, right - 6), left + 6)
        py = max(min(py, bottom - 6), top + 18)
        points.append((px, py))
    if len(points) >= 2:
        draw.line(points, fill=(42, 157, 143), width=max(2, width // 480))
    for px, py in points:
        r = max(2, width // 360)
        draw.ellipse((px - r, py - r, px + r, py + r), fill=(244, 162, 97))
    label = "model waypoints"
    bbox = draw.textbbox((left + 8, top + 6), label, font=font)
    draw.rectangle((bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2), fill=(0, 0, 0))
    draw.text((left + 8, top + 6), label, fill=(255, 255, 255), font=font)


def _build_carla_route_features(
    *,
    transform: Any,
    speed_mps: float,
    route_points: Sequence[Mapping[str, float]],
) -> Tuple[List[float], float]:
    location = transform.location
    progress, _, segment_idx = project_to_route((float(location.x), float(location.y)), route_points)
    command = _route_command_near(route_points, segment_idx)
    near = _sample_route_at_progress(route_points, progress + 5.0)
    target_lookahead_m = 24.0 if int(command) in {1, 2, 3} else 15.0
    target = _sample_route_at_progress(route_points, progress + target_lookahead_m)
    target_local = _world_to_ego_xy(transform, float(target["x"]), float(target["y"]))
    near_local = _world_to_ego_xy(transform, float(near["x"]), float(near["y"]))
    target_model = _carla_forward_right_to_bench2drive_xy(*target_local)
    near_model = _carla_forward_right_to_bench2drive_xy(*near_local)
    return (
        [
            target_model[0] / 50.0,
            target_model[1] / 50.0,
            near_model[0] / 50.0,
            near_model[1] / 50.0,
            float(speed_mps) / 20.0,
            command / 6.0,
            command / 6.0,
            command / 6.0,
        ],
        command,
    )


def _route_command_near(route_points: Sequence[Mapping[str, float]], segment_idx: int) -> float:
    if not route_points:
        return 4.0
    start = max(int(segment_idx), 0)
    current_command = float(route_points[min(start, len(route_points) - 1)].get("command") or 4.0)
    cumulative = 0.0
    previous = route_points[start]
    for point in route_points[start : min(start + 20, len(route_points))]:
        command = float(point.get("command") or 4.0)
        if int(command) in {1, 2, 3}:
            # Do not expose a turn command too early; the vision planner otherwise starts turning before the junction.
            return command if cumulative <= 6.0 else 4.0
        cumulative += math.hypot(float(point["x"]) - float(previous["x"]), float(point["y"]) - float(previous["y"]))
        previous = point
    for point in route_points[start : min(start + 20, len(route_points))]:
        command = float(point.get("command") or 4.0)
        if int(command) in {5, 6}:
            return command if int(current_command) in {5, 6} else 4.0
    return current_command


def _world_to_ego_xy(transform: Any, x: float, y: float) -> Tuple[float, float]:
    yaw = math.radians(float(transform.rotation.yaw))
    dx = float(x) - float(transform.location.x)
    dy = float(y) - float(transform.location.y)
    return math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy


def _carla_forward_right_to_bench2drive_xy(forward_m: float, right_m: float) -> Tuple[float, float]:
    return float(right_m), -float(forward_m)


def _ego_to_world_xy(transform: Any, forward_m: float, right_m: float) -> Tuple[float, float]:
    yaw = math.radians(float(transform.rotation.yaw))
    return (
        float(transform.location.x) + math.cos(yaw) * float(forward_m) - math.sin(yaw) * float(right_m),
        float(transform.location.y) + math.sin(yaw) * float(forward_m) + math.cos(yaw) * float(right_m),
    )


def _bench2drive_waypoint_to_forward_right(point: Sequence[float]) -> Tuple[float, float]:
    return -float(point[1]), float(point[0])


def _carla_state_from_actor(ego: Any) -> Dict[str, float]:
    transform = ego.get_transform()
    velocity = ego.get_velocity()
    speed = math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z)
    return {
        "x": float(transform.location.x),
        "y": float(transform.location.y),
        "yaw": math.radians(float(transform.rotation.yaw)),
        "speed_mps": speed,
    }


def _carla_control_from_prediction(
    *,
    state: Mapping[str, float],
    transform: Any,
    speed_mps: float,
    pred_waypoints: Sequence[Sequence[float]],
    pred_control: Sequence[float],
    brake_probability: float,
    config: CarlaVisionClosedLoopConfig,
    controller_config: PurePursuitConfig,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    target_speed = _target_speed_from_prediction(
        pred_waypoints=pred_waypoints,
        brake_probability=brake_probability,
        speed_mps=speed_mps,
        config=config,
    )
    model_route = _prediction_to_world_route(transform, pred_waypoints)
    if len(model_route) >= 2 and _route_length(model_route) >= 0.5:
        control = pure_pursuit_control(
            state=state,
            route=model_route,
            target_speed_mps=target_speed,
            config=controller_config,
        )
        ego_control_mode = "e2e_waypoint_control"
    else:
        speed_error = float(target_speed) - float(speed_mps)
        throttle = max(min(0.25 * speed_error, 0.45), 0.0)
        brake = max(min(-0.20 * speed_error, 1.0), 0.0)
        control = {"steer": 0.0, "throttle": throttle, "brake": brake}
        ego_control_mode = "e2e_direct"
    if len(pred_control) >= 3:
        model_steer = max(min(float(pred_control[0]), 1.0), -1.0)
        model_throttle = max(min(float(pred_control[1]), 1.0), 0.0)
        model_brake = max(min(float(pred_control[2]), 1.0), 0.0)
        if brake_probability >= float(config.brake_probability_threshold):
            model_throttle = 0.0
            model_brake = max(model_brake, brake_probability)
        else:
            model_brake = 0.0
        control["steer"] = 0.85 * float(control["steer"]) + 0.15 * model_steer
        control["throttle"] = 0.80 * float(control["throttle"]) + 0.20 * model_throttle
        control["brake"] = max(0.80 * float(control["brake"]) + 0.20 * model_brake, 0.0)
    control["steer"] = _stabilize_straight_model_steer(
        steer=float(control["steer"]),
        pred_waypoints=pred_waypoints,
    )
    control = {
        "throttle": max(min(float(control["throttle"]), 0.65), 0.0),
        "steer": max(min(float(control["steer"]), 0.85), -0.85),
        "brake": max(min(float(control["brake"]), 1.0), 0.0),
    }
    if control["brake"] > 0.05:
        control["throttle"] *= max(0.0, 1.0 - control["brake"])
    if float(speed_mps) < 0.20 and float(target_speed) > 0.75 and control["brake"] <= 0.05:
        control["throttle"] = max(float(control["throttle"]), 0.36)
    elif float(speed_mps) < 0.70 and float(target_speed) > 1.5 and control["brake"] <= 0.05:
        control["throttle"] = max(float(control["throttle"]), 0.30)
    return control, {
        "target_speed_mps": target_speed,
        "model_waypoint_controller_weight": 1.0 if ego_control_mode == "e2e_waypoint_control" else 0.0,
        "network_control_blend_weight": 1.0 if len(pred_control) >= 3 else 0.0,
        "control_source": (
            "predicted_waypoints_to_pure_pursuit_plus_control_head"
            if ego_control_mode == "e2e_waypoint_control"
            else "control_head_speed_fallback"
        ),
        "navigation_route_offset_m": 0.0,
        "ego_control_mode": ego_control_mode,
    }


def _stabilize_straight_model_steer(*, steer: float, pred_waypoints: Sequence[Sequence[float]]) -> float:
    if not pred_waypoints:
        return float(steer)
    converted = [_bench2drive_waypoint_to_forward_right(point) for point in pred_waypoints if len(point) >= 2]
    if not converted:
        return float(steer)
    mean_abs_right = mean(abs(float(point[1])) for point in converted)
    final_right = abs(float(converted[-1][1]))
    if max(mean_abs_right, final_right) > MODEL_STRAIGHT_PATH_RIGHT_M:
        return float(steer)
    if abs(float(steer)) <= MODEL_STEER_DEADBAND:
        return 0.0
    if abs(float(steer)) <= MODEL_STEER_DAMPING_BAND:
        return float(steer) * 0.35
    return float(steer)


def _apply_behavior_override(
    *,
    runtime: CarlaScenarioRuntime,
    control: Mapping[str, float],
) -> Tuple[Dict[str, float], Dict[str, str]]:
    del runtime
    updated = dict(control)
    return updated, {"behavior_override": ""}


def _apply_lane_departure_guard(
    *,
    carla: Any,
    carla_map: Any,
    ego: Any,
    route_points: Sequence[Mapping[str, float]],
    control: Mapping[str, float],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    if not route_points:
        return dict(control), {"behavior_override": ""}
    transform = ego.get_transform()
    location = transform.location
    try:
        projected = carla_map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
    except Exception:
        try:
            projected = carla_map.get_waypoint(location, project_to_road=True)
        except Exception:
            projected = None
    if projected is None:
        return dict(control), {"behavior_override": ""}

    progress, lateral_error, _ = project_to_route((float(location.x), float(location.y)), route_points)
    del progress
    try:
        lane_width = float(getattr(projected, "lane_width", 0.0) or 0.0)
        center_distance = float(location.distance(projected.transform.location))
    except Exception:
        lane_width = 0.0
        center_distance = 0.0
    boundary_distance = max(lane_width * 0.42, 1.35) if lane_width > 0.0 else 1.45
    off_route_side = abs(float(lateral_error)) > 2.15
    near_lane_boundary = center_distance > boundary_distance
    steer = float(control.get("steer") or 0.0)
    steering_outward = (float(lateral_error) > 0.0 and steer > 0.03) or (float(lateral_error) < 0.0 and steer < -0.03)
    if not near_lane_boundary and not (off_route_side and steering_outward):
        return dict(control), {"behavior_override": ""}

    target_location = projected.transform.location
    try:
        next_waypoints = list(projected.next(7.0) or [])
        if next_waypoints:
            target_location = next_waypoints[0].transform.location
    except Exception:
        pass
    forward_m, right_m = _world_to_ego_xy(transform, float(target_location.x), float(target_location.y))
    if forward_m < 1.0:
        center_forward, center_right = _world_to_ego_xy(
            transform,
            float(projected.transform.location.x),
            float(projected.transform.location.y),
        )
        forward_m = max(center_forward + 4.0, 1.0)
        right_m = center_right
    desired_steer = max(min(1.15 * math.atan2(float(right_m), max(float(forward_m), 1.0)), 0.42), -0.42)
    guard_weight = 0.72 if near_lane_boundary else 0.45
    updated = dict(control)
    updated["steer"] = max(
        min((1.0 - guard_weight) * steer + guard_weight * desired_steer, 0.50),
        -0.50,
    )
    if near_lane_boundary:
        updated["throttle"] = min(float(updated.get("throttle") or 0.0), 0.24)
    return updated, {
        "behavior_override": "lane_departure_guard",
        "lane_guard_center_distance_m": float(center_distance),
        "lane_guard_route_lateral_m": float(lateral_error),
        "lane_guard_target_steer": float(desired_steer),
    }


def _apply_scenario_safety_override(
    *,
    carla: Any,
    ego: Any,
    route_points: Sequence[Mapping[str, float]],
    runtime: CarlaScenarioRuntime,
    control: Mapping[str, float],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    if not runtime.actors or not route_points:
        return dict(control), {"safety_brake": 0.0}
    ego_location = ego.get_location()
    ego_speed = _actor_speed_mps(ego)
    ego_progress, ego_lateral, _ = project_to_route((float(ego_location.x), float(ego_location.y)), route_points)
    ambient_safety_mode = str(runtime.metadata.get("ambient_safety_mode") or "standard").strip().lower()
    specs = {
        int(dict(spec).get("actor_index") or 0): dict(spec)
        for spec in list(runtime.metadata.get("actor_specs") or [])
    }
    best: Optional[Dict[str, Any]] = None
    for idx, actor in enumerate(runtime.actors):
        try:
            location = actor.get_location()
        except Exception:
            continue
        actor_progress, actor_lateral, _ = project_to_route((float(location.x), float(location.y)), route_points)
        lateral_gap = float(actor_lateral) - float(ego_lateral)
        gap = float(actor_progress) - float(ego_progress)
        spec = specs.get(idx, {})
        kind = str(spec.get("kind") or "")
        role = str(spec.get("role") or "")
        distance = float(ego_location.distance(location))
        active_event_text = str(runtime.metadata.get("active_event") or "")
        crossing_active = kind == "walker" and "pedestrian_crossing" in active_event_text
        crossing_complete = bool(runtime.metadata.get("pedestrian_completed"))
        walker_completion = 1.0
        walker_distance_trigger = False
        walker_path_relevant = False
        walker_path_corridor_m = 0.0
        walker_approach_corridor_m = 0.0
        if kind == "walker":
            target_distance = float(
                spec.get("crosswalk_target_distance_m")
                or runtime.metadata.get("crosswalk_target_distance_m")
                or 8.0
            )
            progress_metadata = spec or runtime.metadata
            has_crossing_progress_metadata = any(
                progress_metadata.get(key) is not None
                for key in ("start_x", "start_y", "x", "y", "direction_x", "direction_y")
            )
            walker_completion = (
                max(
                    min(_walker_crossing_progress(actor, progress_metadata) / max(target_distance, 1e-6), 1.0),
                    0.0,
                )
                if has_crossing_progress_metadata
                else 0.0
            )
            pedestrian_group_triggered = bool(runtime.metadata.get("pedestrian_group_triggered"))
            pedestrian_started = bool(runtime.metadata.get("pedestrian_started"))
            walker_event_relevant = (
                crossing_active
                or pedestrian_group_triggered
                or pedestrian_started
                or walker_completion > 0.02
            )
            pretrigger_close = (
                not walker_event_relevant
                and distance <= 6.0
                and 0.0 <= gap <= 5.0
                and abs(float(lateral_gap)) <= 3.0
            )
            walker_path_corridor_m = max(
                float(
                    spec.get("ego_path_corridor_m")
                    or runtime.metadata.get("pedestrian_ego_path_corridor_m")
                    or 2.8
                ),
                1.8,
            )
            walker_approach_corridor_m = walker_path_corridor_m + max(
                float(
                    spec.get("ego_path_approach_margin_m")
                    or runtime.metadata.get("pedestrian_ego_path_approach_margin_m")
                    or 1.0
                ),
                0.0,
            )
            trigger_distance_m = max(
                float(
                    spec.get("yield_trigger_distance_m")
                    or runtime.metadata.get("pedestrian_yield_trigger_distance_m")
                    or 9.0
                ),
                4.0,
            )
            trigger_gap_m = max(
                float(
                    spec.get("yield_trigger_gap_m")
                    or runtime.metadata.get("pedestrian_yield_trigger_gap_m")
                    or 9.0
                ),
                4.0,
            )
            start_lateral_gap: Optional[float] = None
            if progress_metadata.get("start_x") is not None or progress_metadata.get("start_y") is not None:
                start_x = float(
                    progress_metadata.get("start_x")
                    if progress_metadata.get("start_x") is not None
                    else progress_metadata.get("x")
                    or location.x
                )
                start_y = float(
                    progress_metadata.get("start_y")
                    if progress_metadata.get("start_y") is not None
                    else progress_metadata.get("y")
                    or location.y
                )
                _, start_lateral, _ = project_to_route((start_x, start_y), route_points)
                start_lateral_gap = float(start_lateral) - float(ego_lateral)
            same_side_as_start = (
                start_lateral_gap is not None
                and abs(start_lateral_gap) > 1e-3
                and float(start_lateral_gap) * float(lateral_gap) > 0.0
            )
            inside_ego_path = abs(float(lateral_gap)) <= walker_path_corridor_m
            approaching_ego_path = same_side_as_start and abs(float(lateral_gap)) <= walker_approach_corridor_m
            walker_path_relevant = bool(inside_ego_path or approaching_ego_path or pretrigger_close)
            close_enough_to_yield = bool(distance <= trigger_distance_m or 0.0 <= gap <= trigger_gap_m)
            walker_distance_trigger = bool(
                (
                    walker_event_relevant
                    and not crossing_complete
                    and walker_completion < 0.92
                    and close_enough_to_yield
                    and walker_path_relevant
                )
                or pretrigger_close
            )
        if kind == "walker" and crossing_complete and distance > 4.5:
            continue
        if gap <= 0.0 and not (kind == "walker" and walker_distance_trigger):
            continue
        if role == "ambient_autopilot_vehicle":
            if ambient_safety_mode in {"emergency_only", "minimal", "e2e"}:
                max_gap = 7.0
                corridor_m = 1.35
            else:
                max_gap = 16.0
                corridor_m = 1.65
        elif kind == "walker":
            max_gap = 24.0
            corridor_m = walker_path_corridor_m or 2.8
        else:
            max_gap = 14.0
            corridor_m = 2.2
        if kind == "walker" and not walker_distance_trigger:
            continue
        if not walker_distance_trigger and (gap > max_gap or abs(float(lateral_gap)) > corridor_m):
            continue
        actor_speed = _actor_speed_mps(actor)
        relative_speed = float(ego_speed) - float(actor_speed)
        if role == "ambient_autopilot_vehicle" and not _ambient_vehicle_requires_safety_brake(
            gap_m=gap,
            relative_speed_mps=relative_speed,
            mode=ambient_safety_mode,
        ):
            continue
        if role != "ambient_autopilot_vehicle" and kind != "walker" and not _scenario_vehicle_requires_safety_brake(
            gap_m=gap,
            relative_speed_mps=relative_speed,
        ):
            continue
        priority = (max_gap - gap) + (8.0 if kind == "walker" else 0.0)
        row = {
            "priority": priority,
            "gap_m": gap,
            "distance_m": distance,
            "lateral_m": float(lateral_gap),
            "kind": kind,
            "role": role,
            "relative_speed_mps": relative_speed,
            "walker_completion": walker_completion,
            "walker_path_relevant": walker_path_relevant,
            "walker_path_corridor_m": walker_path_corridor_m,
            "walker_approach_corridor_m": walker_approach_corridor_m,
        }
        if best is None or row["priority"] > float(best["priority"]):
            best = row
    if best is None:
        return dict(control), {"safety_brake": 0.0}

    gap = float(best["gap_m"])
    kind = str(best.get("kind") or "")
    role = str(best.get("role") or "")
    relative_speed = float(best.get("relative_speed_mps") or 0.0)
    distance = float(best.get("distance_m") or -1.0)
    if role == "ambient_autopilot_vehicle":
        if ambient_safety_mode in {"emergency_only", "minimal", "e2e"}:
            if gap < 2.8:
                brake = 1.0
            elif gap < 4.5 and relative_speed > 0.8:
                brake = 0.55
            elif gap < 7.0 and relative_speed > 2.2:
                brake = 0.25
            else:
                brake = 0.0
        else:
            if gap < 5.0:
                brake = 1.0
            elif gap < 8.0 and relative_speed > -0.5:
                brake = 0.80
            elif gap < 12.0 and relative_speed > 0.25:
                brake = 0.55
            elif gap < 16.0 and relative_speed > 1.2:
                brake = 0.30
            else:
                brake = 0.0
    elif kind == "walker":
        crossing_active = "pedestrian_crossing" in str(runtime.metadata.get("active_event") or "")
        crossing_complete = bool(runtime.metadata.get("pedestrian_completed"))
        walker_completion = float(
            best.get("walker_completion") if best.get("walker_completion") is not None else 1.0
        )
        if crossing_active and not crossing_complete and walker_completion < 0.92:
            brake = 0.90 if distance <= 5.5 or gap <= 4.5 else 0.45 if distance <= 8.0 or gap <= 7.0 else 0.18
        elif abs(float(best.get("lateral_m") or 0.0)) > 2.4 and distance > 8.0:
            brake = 0.0
        else:
            brake = 0.80 if gap < 4.5 else 0.35 if gap < 7.0 else 0.0
    else:
        brake = 0.85 if gap < 3.5 else 0.45 if gap < 5.0 else 0.25 if gap < 8.0 else 0.15
    if brake <= 0.0:
        return dict(control), {
            "safety_brake": 0.0,
            "actor_role": str(best.get("role") or ""),
            "gap_m": gap,
            "distance_m": float(best.get("distance_m") or -1.0),
            "lateral_m": float(best.get("lateral_m") or 0.0),
        }
    updated = dict(control)
    updated["throttle"] = 0.0
    updated["brake"] = max(float(updated.get("brake") or 0.0), brake)
    return updated, {
        "safety_brake": updated["brake"],
        "actor_role": str(best.get("role") or ""),
        "gap_m": gap,
        "distance_m": float(best.get("distance_m") or -1.0),
        "lateral_m": float(best.get("lateral_m") or 0.0),
    }


def _ambient_vehicle_requires_safety_brake(*, gap_m: float, relative_speed_mps: float, mode: str = "standard") -> bool:
    if str(mode or "").strip().lower() in {"emergency_only", "minimal", "e2e"}:
        if float(gap_m) < 2.8:
            return True
        if float(gap_m) < 4.5 and float(relative_speed_mps) > 0.8:
            return True
        if float(gap_m) < 7.0 and float(relative_speed_mps) > 2.2:
            return True
        return False
    if float(gap_m) < 5.0:
        return True
    if float(gap_m) < 8.0 and float(relative_speed_mps) > -0.5:
        return True
    if float(gap_m) < 12.0 and float(relative_speed_mps) > 0.25:
        return True
    if float(gap_m) < 16.0 and float(relative_speed_mps) > 1.2:
        return True
    return False


def _scenario_vehicle_requires_safety_brake(*, gap_m: float, relative_speed_mps: float) -> bool:
    if float(gap_m) < 3.5:
        return True
    if float(gap_m) < 5.0 and float(relative_speed_mps) > 0.0:
        return True
    if float(gap_m) < 8.0 and float(relative_speed_mps) > 0.8:
        return True
    if float(gap_m) < 12.0 and float(relative_speed_mps) > 1.8:
        return True
    return False


def _prediction_path_stats(pred_waypoints: Sequence[Sequence[float]]) -> Dict[str, float]:
    converted = []
    for point in pred_waypoints:
        if len(point) < 2:
            continue
        converted.append(_bench2drive_waypoint_to_forward_right(point))
    if not converted:
        return {
            "count": 0.0,
            "path_length_m": 0.0,
            "final_forward_m": 0.0,
            "final_right_m": 0.0,
            "mean_abs_right_m": 0.0,
        }
    path_length = 0.0
    previous = (0.0, 0.0)
    for point in converted:
        path_length += math.hypot(float(point[0]) - previous[0], float(point[1]) - previous[1])
        previous = (float(point[0]), float(point[1]))
    return {
        "count": float(len(converted)),
        "path_length_m": path_length,
        "final_forward_m": float(converted[-1][0]),
        "final_right_m": float(converted[-1][1]),
        "mean_abs_right_m": mean(abs(float(point[1])) for point in converted),
    }


def _prediction_to_world_route(transform: Any, pred_waypoints: Sequence[Sequence[float]]) -> List[Dict[str, float]]:
    route = [{"x": float(transform.location.x), "y": float(transform.location.y)}]
    for point in pred_waypoints:
        forward_m, right_m = _bench2drive_waypoint_to_forward_right(point)
        x, y = _ego_to_world_xy(transform, forward_m, right_m)
        route.append({"x": x, "y": y})
    return route


def _target_speed_from_prediction(
    *,
    pred_waypoints: Sequence[Sequence[float]],
    brake_probability: float,
    speed_mps: float,
    config: CarlaVisionClosedLoopConfig,
) -> float:
    if brake_probability >= float(config.brake_probability_threshold):
        return 0.0
    if len(pred_waypoints) >= 2:
        converted = [_bench2drive_waypoint_to_forward_right(point) for point in pred_waypoints]
        distances = [
            math.hypot(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
            for start, end in zip(converted[:-1], converted[1:])
        ]
        if distances:
            model_speed = mean(distances) / BENCH2DRIVE_DT_S
            if math.isfinite(model_speed):
                blended = 0.70 * model_speed + 0.30 * float(config.target_speed_mps)
                return max(min(blended, float(config.target_speed_mps)), 0.0)
    return float(config.target_speed_mps)


def _smoothstep(value: float) -> float:
    x = max(min(float(value), 1.0), 0.0)
    return x * x * (3.0 - 2.0 * x)


def _sample_route_with_lateral_offset(
    route_points: Sequence[Mapping[str, float]],
    progress_m: float,
    lateral_offset_m: float,
) -> Dict[str, float]:
    point = dict(_sample_route_at_progress(route_points, progress_m))
    offset = float(lateral_offset_m)
    if abs(offset) < 1e-6:
        return point
    yaw = point.get("yaw")
    if yaw is None:
        ahead = _sample_route_at_progress(route_points, float(progress_m) + 2.0)
        yaw = math.atan2(float(ahead["y"]) - float(point["y"]), float(ahead["x"]) - float(point["x"]))
    point["x"] = float(point["x"]) - math.sin(float(yaw)) * offset
    point["y"] = float(point["y"]) + math.cos(float(yaw)) * offset
    point["yaw"] = float(yaw)
    return point


def _sample_route_at_progress(route: Sequence[Mapping[str, float]], progress_m: float) -> Mapping[str, float]:
    if not route:
        return {"x": 0.0, "y": 0.0, "command": 4.0}
    if len(route) == 1:
        return route[0]
    cumulative = 0.0
    for start, end in zip(route[:-1], route[1:]):
        sx, sy = float(start["x"]), float(start["y"])
        ex, ey = float(end["x"]), float(end["y"])
        seg_len = math.hypot(ex - sx, ey - sy)
        if progress_m <= cumulative + seg_len:
            ratio = max(min((float(progress_m) - cumulative) / max(seg_len, 1e-6), 1.0), 0.0)
            return {
                "x": sx + ratio * (ex - sx),
                "y": sy + ratio * (ey - sy),
                "command": float(start.get("command") or end.get("command") or 4.0),
            }
        cumulative += seg_len
    return route[-1]


def _route_length(route: Sequence[Mapping[str, float]]) -> float:
    return sum(
        math.hypot(float(end["x"]) - float(start["x"]), float(end["y"]) - float(start["y"]))
        for start, end in zip(route[:-1], route[1:])
    )


def _evaluate_carla_vision_rollout(
    *,
    states: Sequence[Mapping[str, Any]],
    route_points: Sequence[Mapping[str, float]],
    collision_events: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    route_length = _route_length(route_points)
    completion = max((float(row.get("route_progress_m") or 0.0) for row in states), default=0.0) / max(route_length, 1e-6)
    completion = max(min(completion, 1.0), 0.0)
    lateral_errors = [abs(float(row.get("lateral_error_m") or 0.0)) for row in states]
    offset_adjusted_lateral_errors = [
        abs(float(row.get("lateral_error_m") or 0.0) - float(row.get("navigation_route_offset_m") or 0.0))
        for row in states
    ]
    scenario_distances = [
        float(row.get("scenario_actor_distance_m") or -1.0)
        for row in states
        if float(row.get("scenario_actor_distance_m") or -1.0) >= 0.0
    ]
    predicted_path_lengths = [float(row.get("pred_path_length_m") or 0.0) for row in states]
    predicted_final_forward = [float(row.get("pred_final_forward_m") or 0.0) for row in states]
    predicted_abs_right = [float(row.get("pred_mean_abs_right_m") or 0.0) for row in states]
    walker_completion = [float(row.get("walker_crossing_completion") or 0.0) for row in states]
    walker_lateral_values = [
        float(row.get("walker_route_lateral_m") or 0.0)
        for row in states
        if float(row.get("walker_route_progress_m") or -1.0) >= 0.0
    ]
    post_crossing_speeds = [
        float(row.get("speed_mps") or 0.0)
        for row in states
        if float(row.get("walker_crossing_completion") or 0.0) >= 0.95
    ]
    final_speeds = [float(row.get("speed_mps") or 0.0) for row in list(states)[-20:]]
    right_turn_commands = [
        1.0 if int(float(row.get("command") or 4.0)) == 2 else 0.0
        for row in states
    ]
    comfort_violations = sum(
        1
        for row in states
        if abs(float(row.get("acceleration_mps2") or 0.0)) > 5.5
        or abs(float(row.get("jerk_mps3") or 0.0)) > 15.0
        or abs(float(row.get("yaw_rate_rps") or 0.0)) > 1.3
    )
    lane_failures = sum(1 for value in offset_adjusted_lateral_errors if value > LANE_ERROR_FAILURE_M)
    score = completion
    if collision_events:
        score = 0.0
    score *= max(0.0, 1.0 - 0.05 * lane_failures)
    score *= max(0.75, 1.0 - 0.01 * comfort_violations)
    return {
        "frame_count": len(states),
        "duration_s": float(states[-1]["t_s"]) if states else 0.0,
        "route_length_m": route_length,
        "route_completion": completion,
        "driving_score": max(min(score, 1.0), 0.0),
        "collision_count": len(collision_events),
        "mean_lateral_error_m": mean(lateral_errors) if lateral_errors else 0.0,
        "max_lateral_error_m": max(lateral_errors) if lateral_errors else 0.0,
        "mean_offset_adjusted_lateral_error_m": mean(offset_adjusted_lateral_errors)
        if offset_adjusted_lateral_errors
        else 0.0,
        "max_offset_adjusted_lateral_error_m": max(offset_adjusted_lateral_errors)
        if offset_adjusted_lateral_errors
        else 0.0,
        "comfort_violation_count": comfort_violations,
        "lane_error_failure_count": lane_failures,
        "mean_speed_mps": mean([float(row.get("speed_mps") or 0.0) for row in states]) if states else 0.0,
        "min_scenario_actor_distance_m": min(scenario_distances) if scenario_distances else -1.0,
        "mean_predicted_path_length_m": mean(predicted_path_lengths) if predicted_path_lengths else 0.0,
        "mean_predicted_final_forward_m": mean(predicted_final_forward) if predicted_final_forward else 0.0,
        "mean_predicted_abs_right_m": mean(predicted_abs_right) if predicted_abs_right else 0.0,
        "right_turn_command_ratio": mean(right_turn_commands) if right_turn_commands else 0.0,
        "max_walker_crossing_completion": max(walker_completion) if walker_completion else 0.0,
        "walker_route_lateral_span_m": max(walker_lateral_values) - min(walker_lateral_values)
        if walker_lateral_values
        else 0.0,
        "post_crossing_max_speed_mps": max(post_crossing_speeds) if post_crossing_speeds else 0.0,
        "final_mean_speed_mps": mean(final_speeds) if final_speeds else 0.0,
    }


def _summarize_carla_vision_control_attribution(
    states: Sequence[Mapping[str, Any]],
    runtime: CarlaScenarioRuntime,
) -> Dict[str, Any]:
    frame_count = len(states)
    safety_frames = sum(1 for row in states if float(row.get("safety_brake") or 0.0) > 0.05)
    behavior_override_frames = sum(1 for row in states if str(row.get("behavior_override") or ""))
    navigation_offset_frames = sum(
        1 for row in states if abs(float(row.get("navigation_route_offset_m") or 0.0)) > 1e-3
    )
    traffic_light_conditioned_frames = sum(1 for row in states if float(row.get("traffic_light_conditioned") or 0.0) > 0.5)
    direct_control_frames = sum(
        1 for row in states if str(row.get("ego_control_mode") or "") in {"e2e_waypoint_control", "e2e_direct"}
    )
    model_waypoint_controller_frames = sum(
        1 for row in states if float(row.get("model_waypoint_controller_weight") or 0.0) > 0.5
    )
    network_control_blend_frames = sum(
        1 for row in states if float(row.get("network_control_blend_weight") or 0.0) > 0.5
    )
    lane_guard_frames = sum(1 for row in states if float(row.get("lane_guard_active") or 0.0) > 0.5)
    actor_specs = [dict(item) for item in list(runtime.metadata.get("actor_specs") or [])]
    closed_loop_mode = (
        "vision_model_waypoint_to_pure_pursuit"
        if safety_frames == 0 and behavior_override_frames == 0 and navigation_offset_frames == 0
        else "vision_model_waypoint_control_with_safety_layer"
    )
    return {
        "closed_loop_mode": closed_loop_mode,
        "ego_uses_carla_autopilot": False,
        "ego_uses_map_route_tracking": False,
        "ego_uses_model_waypoints": direct_control_frames > 0,
        "ego_without_safety_override": safety_frames == 0,
        "frame_count": frame_count,
        "model_waypoint_controller_frame_count": model_waypoint_controller_frames,
        "model_waypoint_controller_ratio": model_waypoint_controller_frames / max(frame_count, 1),
        "network_control_blend_frame_count": network_control_blend_frames,
        "network_control_blend_ratio": network_control_blend_frames / max(frame_count, 1),
        "lane_departure_guard_frame_count": lane_guard_frames,
        "lane_departure_guard_ratio": lane_guard_frames / max(frame_count, 1),
        "control_pipeline": "vision_model_waypoints -> pure_pursuit_vehicle_controller -> CARLA VehicleControl",
        "direct_model_control_frame_count": direct_control_frames,
        "direct_model_control_ratio": direct_control_frames / max(frame_count, 1),
        "behavior_override_frame_count": behavior_override_frames,
        "behavior_override_ratio": behavior_override_frames / max(frame_count, 1),
        "safety_override_frame_count": safety_frames,
        "safety_override_ratio": safety_frames / max(frame_count, 1),
        "navigation_offset_frame_count": navigation_offset_frames,
        "navigation_offset_ratio": navigation_offset_frames / max(frame_count, 1),
        "traffic_light_conditioning_frame_count": traffic_light_conditioned_frames,
        "traffic_light_conditioning_ratio": traffic_light_conditioned_frames / max(frame_count, 1),
        "traffic_manager_vehicle_count": sum(
            1 for spec in actor_specs if str(spec.get("role") or "") == "ambient_autopilot_vehicle"
        ),
        "scripted_vehicle_count": sum(
            1
            for spec in actor_specs
            if str(spec.get("kind") or "") == "vehicle"
            and str(spec.get("role") or "") != "ambient_autopilot_vehicle"
        ),
        "crosswalk_walker_count": sum(
            1
            for spec in actor_specs
            if str(spec.get("kind") or "") == "walker" or str(spec.get("placement") or "") == "crosswalk"
        ),
        "controlled_walker_count": sum(1 for spec in actor_specs if str(spec.get("kind") or "") == "walker"),
        "scenario_actor_count": len(actor_specs),
    }


def _write_carla_vision_outputs(
    *,
    output_dir: Path,
    states: Sequence[Mapping[str, Any]],
    frames: Sequence[Any],
    route_points: Sequence[Mapping[str, float]],
    config: CarlaVisionClosedLoopConfig,
    metrics: Mapping[str, Any],
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    states_path = output_dir / "carla_vision_closed_loop_states.csv"
    _write_states_csv(states, states_path)
    route_path = output_dir / "carla_vision_route.json"
    route_path.write_text(json.dumps(list(route_points), indent=2), encoding="utf-8")
    figure_path = output_dir / "carla_vision_closed_loop_rollout.png"
    _plot_carla_vision_rollout(states, route_points, metrics, figure_path)
    media = {
        "states_csv": str(states_path),
        "route_json": str(route_path),
        "figure_path": str(figure_path),
        "gif_path": "",
        "video_path": "",
        "video_encoder": "",
        "contact_sheet_path": "",
    }
    if frames:
        contact_sheet_path = output_dir / "carla_vision_contact_sheet.jpg"
        _write_contact_sheet(frames, states, contact_sheet_path, fps=max(int(config.video_fps), 1))
        media["contact_sheet_path"] = str(contact_sheet_path)
        if bool(config.render_gif):
            gif_path = output_dir / "carla_vision_closed_loop.gif"
            _write_gif(frames, gif_path, fps=max(int(config.video_fps), 1))
            media["gif_path"] = str(gif_path)
        video_path = output_dir / "carla_vision_closed_loop.mp4"
        encoder = _write_mp4(frames, video_path, fps=max(int(config.video_fps), 1), config=config)
        if encoder:
            media["video_path"] = str(video_path)
            media["video_encoder"] = encoder
            media["video_width"] = str(int(frames[0].size[0]))
            media["video_height"] = str(int(frames[0].size[1]))
    return media


def _write_states_csv(states: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "step",
        "frame",
        "t_s",
        "scenario_name",
        "scenario_type",
        "scenario_phase",
        "x",
        "y",
        "z",
        "yaw",
        "speed_mps",
        "acceleration_mps2",
        "jerk_mps3",
        "yaw_rate_rps",
        "route_progress_m",
        "route_completion",
        "lateral_error_m",
        "steer",
        "throttle",
        "brake",
        "target_speed_mps",
        "navigation_route_offset_m",
        "behavior_override",
        "brake_probability",
        "command",
        "ego_control_mode",
        "model_waypoint_controller_weight",
        "network_control_blend_weight",
        "control_source",
        "lane_guard_active",
        "direct_model_control_weight",
        "pred_waypoint_count",
        "pred_path_length_m",
        "pred_final_forward_m",
        "pred_final_right_m",
        "pred_mean_abs_right_m",
        "pred_control_steer",
        "pred_control_throttle",
        "pred_control_brake",
        "pred_waypoints_ego_json",
        "safety_brake",
        "safety_actor_role",
        "safety_gap_m",
        "safety_actor_distance_m",
        "traffic_light_conditioned",
        "ego_route_traffic_light_id",
        "ego_route_traffic_light_group_size",
        "collision_count",
        "scenario_actor_distance_m",
        "scenario_actor_speed_mps",
        "ego_on_driving_lane",
        "ego_nearest_driving_lane_center_distance_m",
        "ego_driving_lane_width_m",
        "natural_traffic_actor_count",
        "natural_traffic_nearest_distance_m",
        "natural_traffic_mean_speed_mps",
        "natural_traffic_visible_actor_count",
        "natural_traffic_nearby_actor_count",
        "natural_traffic_front_actor_count",
        "natural_traffic_adjacent_actor_count",
        "natural_traffic_same_lane_front_actor_count",
        "walker_route_progress_m",
        "walker_route_lateral_m",
        "walker_crossing_progress_m",
        "walker_crossing_target_distance_m",
        "walker_crossing_completion",
    ]
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in states:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _plot_carla_vision_rollout(
    states: Sequence[Mapping[str, Any]],
    route_points: Sequence[Mapping[str, float]],
    metrics: Mapping[str, Any],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=160)
    ax = axes[0]
    if route_points:
        ax.plot([p["x"] for p in route_points], [p["y"] for p in route_points], color="#303030", linewidth=2, label="route")
    if states:
        ax.plot([s["x"] for s in states], [s["y"] for s in states], color="#2a9d8f", linewidth=1.8, label="ego")
        ax.scatter([states[0]["x"]], [states[0]["y"]], color="#2a9d8f", s=28, marker="o", label="start")
        ax.scatter([states[-1]["x"]], [states[-1]["y"]], color="#c44e38", s=28, marker="x", label="end")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    ax.set_title("CARLA route rollout")
    ax = axes[1]
    if states:
        t = [float(s["t_s"]) for s in states]
        ax.plot(t, [float(s["speed_mps"]) for s in states], label="speed m/s")
        ax.plot(t, [abs(float(s["lateral_error_m"])) for s in states], label="|lateral error| m")
        ax.plot(t, [float(s["brake_probability"]) for s in states], label="brake probability")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(
        "completion={0:.3f}, score={1:.3f}".format(
            float(metrics.get("route_completion") or 0.0),
            float(metrics.get("driving_score") or 0.0),
        )
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_gif(frames: Sequence[Any], output_path: Path, *, fps: int) -> None:
    duration_ms = int(round(1000.0 / max(int(fps), 1)))
    first, *rest = list(frames)
    first.save(
        output_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def _write_contact_sheet(frames: Sequence[Any], states: Sequence[Mapping[str, Any]], output_path: Path, *, fps: int) -> None:
    if not frames:
        return
    from PIL import Image, ImageDraw, ImageFont

    sample_count = min(10, len(frames))
    if sample_count <= 1:
        indices = [0]
    else:
        indices = [round(idx * (len(frames) - 1) / (sample_count - 1)) for idx in range(sample_count)]
    tile_width, tile_height = 640, 360
    tiles = []
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for frame_idx in indices:
        state = dict(states[min(int(frame_idx), max(len(states) - 1, 0))]) if states else {}
        image = frames[int(frame_idx)].convert("RGB").resize((tile_width, tile_height))
        draw = ImageDraw.Draw(image)
        t_s = float(state.get("t_s") or (float(frame_idx) / max(int(fps), 1)))
        phase = str(state.get("scenario_phase") or "route_following")
        phase_tokens = [item.strip() for item in phase.split(",") if item.strip()]
        phase = ", ".join(phase_tokens[:2])
        if len(phase_tokens) > 2:
            phase += ", ..."
        completion = float(state.get("route_completion") or 0.0)
        text = f"{t_s:.1f}s  {phase}  completion={completion:.2f}"
        bbox = draw.textbbox((12, 12), text, font=font)
        draw.rectangle((bbox[0] - 6, bbox[1] - 4, bbox[2] + 6, bbox[3] + 4), fill=(0, 0, 0))
        draw.text((12, 12), text, fill=(255, 255, 255), font=font)
        tiles.append(image)
    columns = 2
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), (20, 20, 20))
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % columns) * tile_width, (idx // columns) * tile_height))
    sheet.save(output_path, quality=92)


def _write_mp4(frames: Sequence[Any], output_path: Path, *, fps: int, config: CarlaVisionClosedLoopConfig) -> str:
    preferred_encoder = str(config.video_encoder or "").strip()
    if preferred_encoder:
        if _write_mp4_ffmpeg(
            frames,
            output_path,
            fps=fps,
            encoder=preferred_encoder,
            preset=str(config.video_nvenc_preset or "p4"),
            quality=int(config.video_quality),
        ):
            return preferred_encoder
    if _write_mp4_opencv(frames, output_path, fps=fps):
        return "opencv_mp4v"
    return ""


def _write_mp4_ffmpeg(
    frames: Sequence[Any],
    output_path: Path,
    *,
    fps: int,
    encoder: str,
    preset: str,
    quality: int,
) -> bool:
    ffmpeg_path = shutil.which("ffmpeg") or _env_binary("ffmpeg")
    if not ffmpeg_path or not frames:
        return False
    first = frames[0]
    width, height = first.size
    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{int(width)}x{int(height)}",
        "-r",
        str(max(int(fps), 1)),
        "-i",
        "-",
        "-an",
        "-c:v",
        str(encoder),
    ]
    if str(encoder).endswith("_nvenc"):
        command.extend(["-preset", str(preset or "p4"), "-rc", "vbr", "-cq", str(int(quality)), "-b:v", "0"])
        if str(encoder) == "hevc_nvenc":
            command.extend(["-tag:v", "hvc1"])
    elif str(encoder) == "libx265":
        command.extend(["-crf", str(int(quality)), "-preset", "medium", "-tag:v", "hvc1"])
    command.extend(["-pix_fmt", "yuv420p", str(output_path)])
    try:
        import numpy as np

        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert process.stdin is not None
        for frame in frames:
            rgb = np.asarray(frame.convert("RGB"), dtype=np.uint8)
            process.stdin.write(rgb.tobytes())
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        process.wait(timeout=max(60.0, len(frames) / max(int(fps), 1) * 6.0))
        if process.returncode != 0:
            try:
                (Path(output_path).with_suffix(".ffmpeg.log")).write_bytes(stderr or b"")
            except Exception:
                pass
            return False
    except Exception as exc:
        try:
            (Path(output_path).with_suffix(".ffmpeg.log")).write_text(str(exc), encoding="utf-8")
        except Exception:
            pass
        return False
    return output_path.exists() and output_path.stat().st_size > 0


def _env_binary(name: str) -> str:
    candidate = Path(sys.executable).with_name(name)
    return str(candidate) if candidate.exists() else ""


def _write_mp4_opencv(frames: Sequence[Any], output_path: Path, *, fps: int) -> bool:
    try:
        import cv2
        import numpy as np
    except Exception:
        return False
    if not frames:
        return False
    first = frames[0]
    width, height = first.size
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(int(fps), 1)),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        return False
    for frame in frames:
        rgb = np.asarray(frame.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()
    return output_path.exists() and output_path.stat().st_size > 0


def _write_report_markdown(report: Mapping[str, Any], output_path: Path) -> None:
    metrics = dict(report.get("metrics") or {})
    media = dict(report.get("media") or {})
    scenario = dict(report.get("scenario") or {})
    attribution = dict(report.get("control_attribution") or {})
    lines = [
        "# CARLA Vision Closed-Loop Report",
        "",
        f"- Town: `{report.get('town', '')}`",
        f"- Scenario: `{scenario.get('name', '')}`",
        f"- Scenario type: `{scenario.get('type', '')}`",
        f"- Closed-loop mode: `{attribution.get('closed_loop_mode', '')}`",
        f"- Route length: `{float(report.get('route_length_m') or 0.0):.3f}` m",
        f"- Frames: `{metrics.get('frame_count', 0)}`",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Route completion | `{float(metrics.get('route_completion') or 0.0):.3f}` |",
        f"| Driving score | `{float(metrics.get('driving_score') or 0.0):.3f}` |",
        f"| Mean lateral error | `{float(metrics.get('mean_lateral_error_m') or 0.0):.3f}` m |",
        f"| Mean offset-adjusted lateral error | `{float(metrics.get('mean_offset_adjusted_lateral_error_m') or 0.0):.3f}` m |",
        f"| Collision count | `{metrics.get('collision_count', 0)}` |",
        f"| Mean speed | `{float(metrics.get('mean_speed_mps') or 0.0):.3f}` m/s |",
        f"| Mean predicted path length | `{float(metrics.get('mean_predicted_path_length_m') or 0.0):.3f}` m |",
        f"| Mean predicted final forward distance | `{float(metrics.get('mean_predicted_final_forward_m') or 0.0):.3f}` m |",
        f"| Mean predicted lateral magnitude | `{float(metrics.get('mean_predicted_abs_right_m') or 0.0):.3f}` m |",
        "",
        "| Control Attribution | Value |",
        "| --- | ---: |",
        f"| Ego uses CARLA autopilot | `{bool(attribution.get('ego_uses_carla_autopilot'))}` |",
        f"| Ego uses CARLA route-following controller | `{bool(attribution.get('ego_uses_map_route_tracking'))}` |",
        f"| Ego uses model-predicted waypoints | `{bool(attribution.get('ego_uses_model_waypoints'))}` |",
        f"| Ego without safety override | `{bool(attribution.get('ego_without_safety_override'))}` |",
        f"| Model waypoint-controller frames | `{int(attribution.get('model_waypoint_controller_frame_count') or 0)}` |",
        f"| Model waypoint-controller ratio | `{float(attribution.get('model_waypoint_controller_ratio') or 0.0):.3f}` |",
        f"| Network control-head blend ratio | `{float(attribution.get('network_control_blend_ratio') or 0.0):.3f}` |",
        f"| Lane-departure guard frames | `{int(attribution.get('lane_departure_guard_frame_count') or 0)}` |",
        f"| Behavior override frames | `{int(attribution.get('behavior_override_frame_count') or 0)}` |",
        f"| Safety override frames | `{int(attribution.get('safety_override_frame_count') or 0)}` |",
        f"| Navigation offset frames | `{int(attribution.get('navigation_offset_frame_count') or 0)}` |",
        f"| Traffic-light conditioning frames | `{int(attribution.get('traffic_light_conditioning_frame_count') or 0)}` |",
        f"| Traffic Manager vehicles | `{int(attribution.get('traffic_manager_vehicle_count') or 0)}` |",
        f"| Scripted vehicles | `{int(attribution.get('scripted_vehicle_count') or 0)}` |",
        f"| Crosswalk pedestrians | `{int(attribution.get('crosswalk_walker_count') or 0)}` |",
        f"| Controlled pedestrians | `{int(attribution.get('controlled_walker_count') or 0)}` |",
        "",
    ]
    if media.get("gif_path"):
        lines.append(f"- GIF: `{media.get('gif_path', '')}`")
    if media.get("video_path"):
        video_size = ""
        if media.get("video_width") and media.get("video_height"):
            video_size = f" ({media.get('video_width')}x{media.get('video_height')})"
        lines.append(f"- MP4: `{media.get('video_path', '')}`{video_size}")
    lines.append(f"- States: `{media.get('states_csv', '')}`")
    lines.append("")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def _build_carla_vision_batch_summary(
    *,
    output_dir: Path,
    carla_root: Path,
    checkpoint_path: Path,
    map_name: str,
    requested_town: str,
    reports: Sequence[Mapping[str, Any]],
    started_at: float,
) -> Dict[str, Any]:
    scenario_rows = []
    for report in reports:
        scenario = dict(report.get("scenario") or {})
        metrics = dict(report.get("metrics") or {})
        media = dict(report.get("media") or {})
        attribution = dict(report.get("control_attribution") or {})
        scenario_rows.append(
            {
                "name": str(scenario.get("name") or ""),
                "type": str(scenario.get("type") or ""),
                "actor_count": int(scenario.get("actor_count") or 0),
                "output_dir": str(report.get("output_dir") or ""),
                "metrics": metrics,
                "control_attribution": attribution,
                "media": media,
                "route_length_m": float(report.get("route_length_m") or 0.0),
            }
        )

    completions = [float(row["metrics"].get("route_completion") or 0.0) for row in scenario_rows]
    scores = [float(row["metrics"].get("driving_score") or 0.0) for row in scenario_rows]
    frame_counts = [int(row["metrics"].get("frame_count") or 0) for row in scenario_rows]
    collisions = [int(row["metrics"].get("collision_count") or 0) for row in scenario_rows]
    traffic_manager_counts = [
        int(row["control_attribution"].get("traffic_manager_vehicle_count") or 0) for row in scenario_rows
    ]
    scripted_vehicle_counts = [
        int(row["control_attribution"].get("scripted_vehicle_count") or 0) for row in scenario_rows
    ]
    crosswalk_walker_counts = [
        int(row["control_attribution"].get("crosswalk_walker_count") or 0) for row in scenario_rows
    ]
    summary = {
        "schema": "carla_vision_closed_loop_batch_v1",
        "output_dir": str(output_dir),
        "carla_root": str(carla_root),
        "checkpoint_path": str(checkpoint_path),
        "town": str(map_name),
        "requested_town": str(requested_town or ""),
        "scenario_count": len(scenario_rows),
        "aggregate": {
            "mean_route_completion": mean(completions) if completions else 0.0,
            "mean_driving_score": mean(scores) if scores else 0.0,
            "total_collisions": sum(collisions),
            "total_frames": sum(frame_counts),
            "total_traffic_manager_vehicles": sum(traffic_manager_counts),
            "total_scripted_vehicles": sum(scripted_vehicle_counts),
            "total_crosswalk_pedestrians": sum(crosswalk_walker_counts),
        },
        "scenarios": scenario_rows,
        "runtime_s": round(time.time() - float(started_at), 3),
    }
    return summary


def _write_carla_vision_batch_markdown(summary: Mapping[str, Any], output_path: Path) -> None:
    aggregate = dict(summary.get("aggregate") or {})
    lines = [
        "# CARLA Vision Closed-Loop Batch",
        "",
        f"- Town: `{summary.get('town', '')}`",
        f"- Scenarios: `{summary.get('scenario_count', 0)}`",
        f"- Mean route completion: `{float(aggregate.get('mean_route_completion') or 0.0):.3f}`",
        f"- Mean driving score: `{float(aggregate.get('mean_driving_score') or 0.0):.3f}`",
        f"- Total collisions: `{int(aggregate.get('total_collisions') or 0)}`",
        f"- Traffic Manager vehicles: `{int(aggregate.get('total_traffic_manager_vehicles') or 0)}`",
        f"- Scripted vehicles: `{int(aggregate.get('total_scripted_vehicles') or 0)}`",
        f"- Crosswalk pedestrians: `{int(aggregate.get('total_crosswalk_pedestrians') or 0)}`",
        "",
        "| Scenario | Type | Frames | TM Vehicles | Scripted Vehicles | Pedestrians | Completion | Driving Score | Collisions | Predicted Path | Min actor distance | MP4 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in list(summary.get("scenarios") or []):
        scenario = dict(row)
        metrics = dict(scenario.get("metrics") or {})
        media = dict(scenario.get("media") or {})
        attribution = dict(scenario.get("control_attribution") or {})
        video_path = str(media.get("video_path") or "")
        min_distance = float(metrics.get("min_scenario_actor_distance_m") or -1.0)
        min_distance_text = f"{min_distance:.2f} m" if min_distance >= 0.0 else ""
        lines.append(
            "| {name} | {stype} | {frames} | {tm_vehicles} | {scripted_vehicles} | {pedestrians} | {completion:.3f} | {score:.3f} | {collisions} | {pred_path:.2f} m | {distance} | `{mp4}` |".format(
                name=str(scenario.get("name") or ""),
                stype=str(scenario.get("type") or ""),
                frames=int(metrics.get("frame_count") or 0),
                tm_vehicles=int(attribution.get("traffic_manager_vehicle_count") or 0),
                scripted_vehicles=int(attribution.get("scripted_vehicle_count") or 0),
                pedestrians=int(attribution.get("crosswalk_walker_count") or 0),
                completion=float(metrics.get("route_completion") or 0.0),
                score=float(metrics.get("driving_score") or 0.0),
                collisions=int(metrics.get("collision_count") or 0),
                pred_path=float(metrics.get("mean_predicted_path_length_m") or 0.0),
                distance=min_distance_text,
                mp4=video_path,
            )
        )
    lines.append("")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def _config_to_json(config: CarlaVisionClosedLoopConfig) -> Dict[str, Any]:
    payload = dict(config.__dict__)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def format_carla_vision_launch_command(config: CarlaVisionClosedLoopConfig) -> str:
    command = build_carla_launch_command(
        carla_root=Path(config.carla_root),
        render_offscreen=True,
        null_rhi=False,
        port=int(config.port),
        quality_level=str(config.carla_quality_level or "Epic"),
        fps=int(config.fps),
    )
    return format_carla_launch_command(command, cuda_visible_devices=config.cuda_visible_devices)
