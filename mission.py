#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


REPO_ROOT = Path(__file__).resolve().parent
IMAGES_DIR = REPO_ROOT / "Images"
RUNTIME_DIR = IMAGES_DIR / "_mission_runtime"
RUNTIME_REF_PATH = RUNTIME_DIR / "reference_fullres.png"
LOG_PATH = IMAGES_DIR / "Log.txt"
STATE_PATH = REPO_ROOT / "mission_state.json"
CHANGE_NODE = REPO_ROOT / "change_detect.py"

FLAG_SET_REF = Path("/tmp/change_set_reference")
FLAG_CLEAR_REF = Path("/tmp/change_clear_reference")
FLAG_SAVE = Path("/tmp/change_save_outputs")
FLAG_QUIT = Path("/tmp/change_quit")
LIVE_STATUS = Path("/tmp/change_live/status.txt")
PISUGAR_SOCKET = "/tmp/pisugar-server.sock"

ROI_WIDTH = 1060
ROI_HEIGHT = 1040
SAFE_THRESHOLD = 150
MIN_SAFE_THRESHOLD = 10
MIN_SAFE_DENSITY = 0.5
REFERENCE_DELAY_SEC = 15
PRE_SHUTDOWN_DELAY_SEC = 10
POST_REBOOT_CAPTURE_DELAY_SEC = 15
WAKE_DELAY_SEC = 30
NODE_READY_TIMEOUT_SEC = 45
SAVE_TIMEOUT_SEC = 45
ALARM_REPEAT_NONE = 0
PISUGAR_SOCKET_TIMEOUT_SEC = 2


@dataclass
class MissionState:
    phase: str = "await_reference"
    current_pass: Optional[str] = None
    boot_count: int = 0
    last_boot_utc: Optional[str] = None
    last_action: str = "Mission state initialized"
    last_completed_pass: Optional[str] = None


def require_processing_deps() -> None:
    if cv2 is None or np is None:
        raise RuntimeError("cv2 and numpy are required for mission processing")


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dirs() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.write_text("", encoding="utf-8")


def load_state() -> MissionState:
    if not STATE_PATH.exists():
        return MissionState()
    with STATE_PATH.open("r", encoding="utf-8") as handle:
        return MissionState(**json.load(handle))


def save_state(state: MissionState) -> None:
    temp_path = STATE_PATH.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(state), handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, STATE_PATH)


def append_log(pass_id: Optional[str], message: str) -> None:
    line = f"{f'[{pass_id}] ' if pass_id else ''}{iso_now()} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def set_state(state: MissionState, phase: str, action: str) -> None:
    state.phase = phase
    state.last_action = action
    save_state(state)


def git_run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=check,
        text=True,
        capture_output=True,
    )


def repo_is_dirty() -> bool:
    status = git_run("status", "--porcelain", check=False)
    return bool(status.stdout.strip())


def sync_repo(pass_id: Optional[str], push: bool = False) -> None:
    if not push:
        if repo_is_dirty():
            append_log(pass_id, "Git pull skipped: repo is dirty")
            return
        try:
            git_run("pull", "--rebase")
            append_log(pass_id, "Git pull completed")
        except subprocess.CalledProcessError as exc:
            append_log(pass_id, f"Git pull failed: {exc.stderr.strip() or exc.stdout.strip()}")
        return

    try:
        paths = ["mission_state.json", "Images/Log.txt"]
        if pass_id:
            paths.append(f"Images/{pass_id}")
        git_run("add", *paths)
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", *paths],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if staged.returncode == 0:
            append_log(pass_id, "Git push skipped: no changes")
            return
        git_run("commit", "-m", f"Mission update {pass_id or iso_now()}")
        push_result = git_run("push", check=False)
        if push_result.returncode != 0:
            append_log(pass_id, f"Git push retry after rebase: {push_result.stderr.strip() or push_result.stdout.strip()}")
            git_run("pull", "--rebase")
            git_run("push")
        append_log(pass_id, "Git push completed")
    except subprocess.CalledProcessError as exc:
        append_log(pass_id, f"Git push failed: {exc.stderr.strip() or exc.stdout.strip()}")


def next_pass_id() -> str:
    highest = 0
    for item in IMAGES_DIR.iterdir():
        if item.is_dir() and len(item.name) == 4 and item.name.startswith("P") and item.name[1:].isdigit():
            highest = max(highest, int(item.name[1:]))
    return f"P{highest + 1:03d}"


def pass_dir(pass_id: str) -> Path:
    path = IMAGES_DIR / pass_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def touch_flag(path: Path) -> None:
    path.write_text("", encoding="utf-8")


def wait_for_reference_capture() -> None:
    previous_mtime = RUNTIME_REF_PATH.stat().st_mtime if RUNTIME_REF_PATH.exists() else None

    touch_flag(FLAG_SET_REF)

    def reference_ready() -> bool:
        if FLAG_SET_REF.exists():
            return False
        if not RUNTIME_REF_PATH.exists():
            return False

        current_mtime = RUNTIME_REF_PATH.stat().st_mtime
        if previous_mtime is not None and current_mtime <= previous_mtime:
            return False

        if cv2 is None:
            return True
        return cv2.imread(str(RUNTIME_REF_PATH), cv2.IMREAD_COLOR) is not None

    wait_for(reference_ready, NODE_READY_TIMEOUT_SEC, "reference image capture")


def wait_for(predicate, timeout_sec: int, label: str) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {label}")


def start_change_node() -> subprocess.Popen:
    if os.name == "nt":
        raise RuntimeError("mission.py must run on the Raspberry Pi, not Windows")
    if not CHANGE_NODE.exists():
        raise RuntimeError(f"Change detect node not found at {CHANGE_NODE}")
    process = subprocess.Popen([sys.executable, str(CHANGE_NODE)], cwd=REPO_ROOT)
    wait_for(lambda: LIVE_STATUS.exists(), NODE_READY_TIMEOUT_SEC, "change-detect node readiness")
    return process


def stop_change_node(process: Optional[subprocess.Popen]) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        touch_flag(FLAG_QUIT)
        process.wait(timeout=10)
    except Exception:
        process.terminate()
        process.wait(timeout=5)


def list_timestamp_dirs() -> set[str]:
    return {item.name for item in IMAGES_DIR.iterdir() if item.is_dir() and item.name[:8].isdigit()}


def wait_for_new_bundle(existing: set[str]) -> Path:
    required = {"mask.png", "ref_fullres.png", "cur_fullres.png"}

    def bundle_ready(candidate: Path) -> bool:
        present = {path.name for path in candidate.iterdir() if path.is_file()}
        if not required.issubset(present):
            return False
        if cv2 is None:
            return True
        mask = cv2.imread(str(candidate / "mask.png"), cv2.IMREAD_GRAYSCALE)
        ref_fullres = cv2.imread(str(candidate / "ref_fullres.png"), cv2.IMREAD_COLOR)
        cur_fullres = cv2.imread(str(candidate / "cur_fullres.png"), cv2.IMREAD_COLOR)
        return mask is not None and ref_fullres is not None and cur_fullres is not None

    def bundle_created() -> bool:
        for name in sorted(list_timestamp_dirs() - existing):
            candidate = IMAGES_DIR / name
            if bundle_ready(candidate):
                return True
        return False

    wait_for(bundle_created, SAVE_TIMEOUT_SEC, "change-detect save bundle")
    for name in sorted(list_timestamp_dirs() - existing, reverse=True):
        candidate = IMAGES_DIR / name
        if bundle_ready(candidate):
            return candidate
    raise RuntimeError("Change-detect bundle directory appeared, but required files were not ready")


def largest_component(mask: np.ndarray) -> Optional[dict]:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = None
    best_area = 0
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best = {
                "x": int(stats[idx, cv2.CC_STAT_LEFT]),
                "y": int(stats[idx, cv2.CC_STAT_TOP]),
                "w": int(stats[idx, cv2.CC_STAT_WIDTH]),
                "h": int(stats[idx, cv2.CC_STAT_HEIGHT]),
                "area": area,
            }
    return best


def clamp_box(x: int, y: int, width: int, height: int, image_w: int, image_h: int) -> tuple[int, int, int, int]:
    x = max(0, min(x, image_w - width))
    y = max(0, min(y, image_h - height))
    return x, y, width, height


def scale_box(box: tuple[int, int, int, int], src_shape: tuple[int, int], dst_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    src_h, src_w = src_shape[:2]
    dst_h, dst_w = dst_shape[:2]
    x, y, w, h = box
    scaled_x = int(round(x * (dst_w / src_w)))
    scaled_y = int(round(y * (dst_h / src_h)))
    scaled_w = max(1, int(round(w * (dst_w / src_w))))
    scaled_h = max(1, int(round(h * (dst_h / src_h))))
    return clamp_box(scaled_x, scaled_y, scaled_w, scaled_h, dst_w, dst_h)


def landing_box_size(image_shape: tuple[int, int, int]) -> tuple[int, int]:
    image_h, image_w = image_shape[:2]
    scale_x = image_w / 4608.0
    scale_y = image_h / 2592.0
    width = min(max(1, int(round(ROI_WIDTH * scale_x))), image_w)
    height = min(max(1, int(round(ROI_HEIGHT * scale_y))), image_h)
    return width, height


def safe_mask_from_gray(gray: np.ndarray, threshold: int = SAFE_THRESHOLD) -> np.ndarray:
    _, safe = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    safe = cv2.morphologyEx(safe, cv2.MORPH_OPEN, kernel)
    safe = cv2.morphologyEx(safe, cv2.MORPH_CLOSE, kernel)
    return safe


def safe_mask_from_image(bgr: np.ndarray, threshold: int = SAFE_THRESHOLD) -> np.ndarray:
    return safe_mask_from_gray(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), threshold)


def landing_zone_details(bgr: np.ndarray) -> dict:
    image_h, image_w = bgr.shape[:2]
    width, height = landing_box_size(bgr.shape)
    window_area = width * height
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    threshold = SAFE_THRESHOLD
    best_attempt = None

    while threshold >= MIN_SAFE_THRESHOLD:
        safe = safe_mask_from_gray(gray, threshold)
        safe_f = (safe > 0).astype(np.float32)
        integral = cv2.integral(safe_f)

        y_slots = image_h - height + 1
        x_slots = image_w - width + 1
        counts = (
            integral[height : height + y_slots, width : width + x_slots]
            - integral[0:y_slots, width : width + x_slots]
            - integral[height : height + y_slots, 0:x_slots]
            + integral[0:y_slots, 0:x_slots]
        )
        max_count = float(counts.max()) if counts.size else 0.0
        flat_idx = int(np.argmax(counts)) if counts.size else 0
        best_y, best_x = np.unravel_index(flat_idx, counts.shape) if counts.size else (0, 0)
        safe_density = max_count / float(window_area) if window_area else 0.0

        attempt = {
            "box": clamp_box(int(best_x), int(best_y), width, height, image_w, image_h),
            "threshold_used": threshold,
            "safe_pixels": int(round(max_count)),
            "safe_density": safe_density,
        }
        if best_attempt is None or attempt["safe_pixels"] > best_attempt["safe_pixels"]:
            best_attempt = attempt

        if max_count > 0 and safe_density >= MIN_SAFE_DENSITY:
            return {
                **attempt,
                "selection_status": "ok",
                "selection_reason": "adaptive safe-area search",
                "minimum_safe_density": MIN_SAFE_DENSITY,
            }

        threshold //= 2

    fallback_box = clamp_box((image_w - width) // 2, (image_h - height) // 2, width, height, image_w, image_h)
    fallback = {
        "box": fallback_box,
        "threshold_used": best_attempt["threshold_used"] if best_attempt is not None else SAFE_THRESHOLD,
        "safe_pixels": best_attempt["safe_pixels"] if best_attempt is not None else 0,
        "safe_density": best_attempt["safe_density"] if best_attempt is not None else 0.0,
        "selection_status": "fallback_center",
        "selection_reason": "no threshold met minimum safe density",
        "minimum_safe_density": MIN_SAFE_DENSITY,
    }
    if best_attempt is not None:
        fallback["best_attempt_box"] = {
            "x": best_attempt["box"][0],
            "y": best_attempt["box"][1],
            "width": best_attempt["box"][2],
            "height": best_attempt["box"][3],
        }
    return fallback


def landing_zone_box(bgr: np.ndarray) -> tuple[int, int, int, int]:
    return landing_zone_details(bgr)["box"]


def scale_component(component: Optional[dict], src_shape: tuple[int, int], dst_shape: tuple[int, int, int]) -> Optional[dict]:
    if component is None:
        return None
    src_h, src_w = src_shape[:2]
    dst_h, dst_w = dst_shape[:2]
    scale_x = dst_w / src_w
    scale_y = dst_h / src_h
    return {
        "x": int(round(component["x"] * scale_x)),
        "y": int(round(component["y"] * scale_y)),
        "w": max(1, int(round(component["w"] * scale_x))),
        "h": max(1, int(round(component["h"] * scale_y))),
        "area": component["area"],
    }


def organize_bundle(pass_id: str, bundle_dir: Path) -> str:
    require_processing_deps()
    folder = pass_dir(pass_id)

    mask = cv2.imread(str(bundle_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    current_fullres = cv2.imread(str(bundle_dir / "cur_fullres.png"), cv2.IMREAD_COLOR)
    if mask is None or current_fullres is None:
        raise RuntimeError(f"Saved bundle in {bundle_dir} is missing required files")

    landing = landing_zone_details(current_fullres)
    x, y, w, h = landing["box"]
    bm_x, bm_y, bm_w, bm_h = scale_box((x, y, w, h), current_fullres.shape[:2], mask.shape[:2])
    if landing["selection_status"] != "ok":
        append_log(
            pass_id,
            f"Landing zone fallback used: {landing['selection_reason']} "
            f"(threshold={landing['threshold_used']}, safe_density={landing['safe_density']:.3f})",
        )

    binary_id = f"BM-{pass_id}"
    shutil.move(str(bundle_dir / "ref_fullres.png"), str(folder / "reference.png"))
    shutil.move(str(bundle_dir / "cur_fullres.png"), str(folder / "current.png"))
    shutil.move(str(bundle_dir / "mask.png"), str(folder / "binary_mask_raw.png"))
    shutil.move(str(bundle_dir / "overlay.png"), str(folder / "overlay.png"))
    shutil.move(str(bundle_dir / "ref.png"), str(folder / "reference_preview.png"))
    shutil.move(str(bundle_dir / "cur.png"), str(folder / "current_preview.png"))
    if (bundle_dir / "meta.txt").exists():
        shutil.move(str(bundle_dir / "meta.txt"), str(folder / "change_detect_meta.txt"))

    boxed_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(boxed_mask, (bm_x, bm_y), (bm_x + bm_w, bm_y + bm_h), (0, 255, 0), 2)
    cv2.imwrite(str(folder / f"{binary_id}.png"), boxed_mask)

    boxed = current_fullres.copy()
    cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 8)
    cv2.imwrite(str(folder / "boxed_fullres.png"), boxed)
    cv2.imwrite(str(folder / "roi.png"), current_fullres[y:y + h, x:x + w])

    meta = {
        "pass_id": pass_id,
        "binary_id": binary_id,
        "bundle_source": bundle_dir.name,
        "generated_utc": iso_now(),
        "roi_box": {"x": x, "y": y, "width": w, "height": h},
        "bm_box": {"x": bm_x, "y": bm_y, "width": bm_w, "height": bm_h},
        "mask_size": {"width": mask.shape[1], "height": mask.shape[0]},
        "image_size": {"width": current_fullres.shape[1], "height": current_fullres.shape[0]},
        "landing_zone_method": "adaptive white-safe-area integral search",
        "landing_threshold_used": landing["threshold_used"],
        "landing_safe_pixels": landing["safe_pixels"],
        "landing_safe_density": landing["safe_density"],
        "landing_minimum_safe_density": landing["minimum_safe_density"],
        "landing_selection_status": landing["selection_status"],
        "landing_selection_reason": landing["selection_reason"],
    }
    if "best_attempt_box" in landing:
        meta["landing_best_attempt_box"] = landing["best_attempt_box"]
    with (folder / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    shutil.rmtree(bundle_dir, ignore_errors=True)
    return binary_id


def clear_reference() -> None:
    touch_flag(FLAG_CLEAR_REF)
    time.sleep(1)


def pisugar_send(command: str) -> str:
    if not os.path.exists(PISUGAR_SOCKET):
        raise RuntimeError(f"PiSugar socket not found at {PISUGAR_SOCKET}")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(PISUGAR_SOCKET_TIMEOUT_SEC)
        client.connect(PISUGAR_SOCKET)
        client.sendall((command.strip() + "\n").encode("utf-8"))
        client.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            try:
                chunk = client.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks).decode("utf-8", errors="replace").strip()


def schedule_wake_and_shutdown(pass_id: str) -> None:
    append_log(pass_id, "Scheduling PiSugar test wake")
    response = pisugar_send("rtc_test_wake")
    append_log(pass_id, f"PiSugar response: {response or 'ok'}")
    append_log(pass_id, "Requesting system shutdown")
    subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)


def start_reference_cycle(state: MissionState) -> None:
    pass_id = next_pass_id()
    state.current_pass = pass_id
    save_state(state)

    append_log(pass_id, "Cubesat in orbit")
    set_state(state, "STANDBY", "Waiting before POI lock")
    time.sleep(REFERENCE_DELAY_SEC)

    append_log(pass_id, "POI locked")
    set_state(state, "POI_LOCKED", "Setting reference image in change-detect node")
    wait_for_reference_capture()
    append_log(pass_id, "Reference image captured")

    time.sleep(PRE_SHUTDOWN_DELAY_SEC)
    set_state(state, "await_post_reboot_capture", "Reference image captured; waiting for reboot cycle")
    append_log(pass_id, "Preparing PiSugar shutdown and reboot")
    schedule_wake_and_shutdown(pass_id)


def resume_post_reboot_cycle(state: MissionState) -> None:
    pass_id = state.current_pass
    if not pass_id:
        raise RuntimeError("State is waiting for post-reboot capture, but no pass is active")

    append_log(pass_id, "Recovery boot detected")
    set_state(state, "REBOOT_RECOVERY", "Waiting before post-reboot save request")
    time.sleep(POST_REBOOT_CAPTURE_DELAY_SEC)

    existing = list_timestamp_dirs()
    touch_flag(FLAG_SAVE)
    bundle_dir = wait_for_new_bundle(existing)
    append_log(pass_id, f"Change-detect bundle saved to {bundle_dir.name}")

    set_state(state, "PROCESSING", "Organizing full-resolution mission outputs")
    binary_id = organize_bundle(pass_id, bundle_dir)
    append_log(pass_id, f"Difference map saved as {binary_id}")
    append_log(pass_id, "Largest white region boxed")

    set_state(state, "DOWNLINK", "Syncing mission artifacts to GitHub")
    sync_repo(pass_id, push=True)
    clear_reference()

    state.phase = "await_reference"
    state.last_completed_pass = pass_id
    state.current_pass = None
    state.last_action = f"Completed {pass_id}"
    save_state(state)


def run_mission() -> None:
    ensure_dirs()
    sync_repo(None, push=False)
    state = load_state()
    state.boot_count += 1
    state.last_boot_utc = iso_now()
    save_state(state)

    node = start_change_node()
    try:
        while True:
            if state.phase == "await_post_reboot_capture":
                resume_post_reboot_cycle(state)
            else:
                start_reference_cycle(state)
    finally:
        stop_change_node(node)


def print_status() -> None:
    ensure_dirs()
    print(json.dumps(asdict(load_state()), indent=2))


def upload_land(pass_id: str) -> None:
    folder = IMAGES_DIR / pass_id
    if not (folder / "boxed_fullres.png").exists():
        raise RuntimeError(f"{folder / 'boxed_fullres.png'} does not exist")
    append_log(pass_id, f"LAND command received for {pass_id}")
    sync_repo(pass_id, push=True)


def upload_binary(binary_id: str) -> None:
    for folder in IMAGES_DIR.iterdir():
        if folder.is_dir() and (folder / f"{binary_id}.png").exists():
            append_log(folder.name, f"Binary upload requested for {binary_id}")
            sync_repo(folder.name, push=True)
            return
    raise RuntimeError(f"Could not find binary map {binary_id}")


def normalize_argv(argv: list[str]) -> list[str]:
    if len(argv) < 2:
        return argv
    mapping = {"RUN": "run", "STATUS": "status", "LAND": "land", "UPLOAD-BINARY": "upload-binary"}
    argv = argv[:]
    argv[1] = mapping.get(argv[1].upper(), argv[1])
    return argv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CubeSat mission controller")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run", help="Run the mission state machine")
    subparsers.add_parser("status", help="Show persisted mission status")

    land_parser = subparsers.add_parser("land", help="Upload artifacts for a specific pass")
    land_parser.add_argument("pass_id", help="Pass identifier, for example P001")

    binary_parser = subparsers.add_parser("upload-binary", help="Upload a specific binary map")
    binary_parser.add_argument("binary_id", help="Binary map identifier, for example BM-P001")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    argv = normalize_argv(argv or sys.argv)
    args = build_parser().parse_args(argv[1:])
    try:
        if args.command == "run":
            run_mission()
        elif args.command == "status":
            print_status()
        elif args.command == "land":
            upload_land(args.pass_id)
        elif args.command == "upload-binary":
            upload_binary(args.binary_id)
    except Exception as exc:
        ensure_dirs()
        append_log(None, f"Mission error: {exc}")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
