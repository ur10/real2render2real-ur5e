"""Build a "Scan, Track, Render" page like real2render2real.com for the UR5e cube demo.

Left: the human RGB-D demo (RealSense front view). Right: an interactive viser recording (same
format/version as the project page, 0.2.23) of the UR5e + 2F-85 replaying the tracked cube motion:
robot URDF animated from a generated rollout's joints, the cube's Gaussian splat moving along the
logged object poses, the phone-scanned bench splat, and both RealSense frustums showing the
rendered camera images.

Run with viser 0.2.23 (nerfstudio pins 0.2.7 in .venv):
    uv run --no-project --with viser==0.2.23 --with yourdfpy --with pyliblzfse --with h5py \
        python scripts/ur_cube/make_viser_recording.py --rollout <successes/env_...> \
        --output-dir outputs/ur_cube/review/scan_track_render
Then serve it:  python3 -m http.server 8090 -d outputs/ur_cube/review/scan_track_render
"""

import json
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import tyro
import viser
import yourdfpy
from PIL import Image
from viser.extras import ViserUrdf

REPO = Path(__file__).resolve().parents[2]
CAMERA_NAMES = ("front", "right")  # order of data/camera_poses/ur_realsense_front_right.h5

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Scan, Track, Render: UR5e cube</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 16px; background: #fafafa; color: #222; }}
 .row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
 .col {{ flex: 1 1 480px; min-width: 320px; }}
 video, iframe {{ width: 100%; aspect-ratio: 16 / 10; border: 1px solid #ccc; border-radius: 6px; background: #000; }}
 p.cap {{ color: #555; font-size: 14px; }}
</style></head><body>
<h2>Scan, Track, Render</h2>
<div class="row">
 <div class="col"><video src="demo.mp4" autoplay loop muted playsinline controls></video>
  <p class="cap">Human demonstration (RealSense front, capture rate 10 Hz): cube moved from the far to the near fixture.</p></div>
 <div class="col"><iframe src="viser-client/index.html?playbackPath=../{recording}&initialCameraPosition={cam_pos}&initialCameraLookAt={cam_look}&initialCameraUp=0,0,1"></iframe>
  <p class="cap">Rendered robot replay (drag to rotate): UR5e + Robotiq 2F-85, the scanned cube's Gaussian splat following the tracked trajectory, the phone-scanned bench, and both RealSense views.</p></div>
</div></body></html>
"""


def load_jpg(path: Path, width: int) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.asarray(im.resize((width, int(im.height * width / im.width))))


def main(
    rollout: Path,
    output_dir: Path,
    splats: Path = REPO / "outputs/ur_cube/splats_base.npz",
    urdf: Path = REPO / "data/ur5e_robotiq/ur5e_robotiq85.urdf",
    camera_poses: Path = REPO / "data/camera_poses/ur_realsense_front_right.h5",
    intrinsics: Path = REPO / "data/camera_poses/ur_realsense_intrinsics.json",
    layout: Path = REPO / "outputs/ur_cube/scene_layout.json",
    demo_rgbd_dir: Path = Path.home() / "ur_human_capture/rgbd/0",
    demo_steps: tuple[int, int] = (52, 84),
    fps: float = 30.0,
    frustum_image_width: int = 320,
    include_bench: bool = True,
    bench_radius: float = 1.8,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sp = np.load(splats)
    table_z = json.loads(layout.read_text())["table_z"]
    intr = json.loads(intrinsics.read_text())
    with h5py.File(camera_poses, "r") as f:
        cam_poses = f["poses"][:]  # (2, 7) xyz + wxyz, OpenCV optical
    with h5py.File(rollout / "robot_data/robot_data.h5", "r") as f:
        joints = f["joint_angles"][:]  # (T, 7): 6 arm + finger_joint
    obj = np.load(rollout / "robot_data/object_poses.npy")  # (T, 7) wxyz_xyz, base frame
    n = min(len(joints), len(obj), *(len(list((rollout / f"camera_{i}/rgb").glob("*.jpg"))) for i in range(2)))
    print(f"rollout {rollout.name}: {n} steps")

    server = viser.ViserServer(port=8155)
    rec = server.get_scene_serializer()
    if include_bench:  # the scanned bench is the ground; drop what's under the table top
        b = (sp["bench_centers"][:, 2] > table_z - 0.03) & (np.linalg.norm(sp["bench_centers"][:, :2] - [0.15, -0.45], axis=1) < bench_radius)
        server.scene.add_gaussian_splats(
            "/bench/gaussian_splat", centers=sp["bench_centers"][b], covariances=sp["bench_covariances"][b],
            rgbs=sp["bench_rgbs"][b], opacities=sp["bench_opacities"][b],
        )
    else:  # object-only view like the project page
        server.scene.add_grid("/ground", width=3.0, height=3.0, position=(0.15, -0.45, table_z - 0.002), cell_size=0.1)
    cube = server.scene.add_frame("/cube", show_axes=False, wxyz=obj[0, :4], position=obj[0, 4:])
    server.scene.add_gaussian_splats(
        "/cube/gaussian_splat", centers=sp["cube_centers"], covariances=sp["cube_covariances"],
        rgbs=sp["cube_rgbs"], opacities=sp["cube_opacities"],
    )
    robot = ViserUrdf(server, yourdfpy.URDF.load(
        urdf, filename_handler=lambda fname: yourdfpy.filename_handler_magic(fname, dir=urdf.parent)
    ), root_node_name="/robot")
    names = robot.get_actuated_joint_names()
    order = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
             "wrist_3_joint", "finger_joint"]
    idx = [order.index(j) for j in names]

    fov = 2 * np.arctan(intr["height"] / 2 / intr["shared_K"][4])
    frustums = []
    for i, name in enumerate(CAMERA_NAMES):
        p = cam_poses[i]
        frustums.append(server.scene.add_camera_frustum(
            f"/realsense_{name}", fov=fov, aspect=intr["width"] / intr["height"], scale=0.05,
            wxyz=p[3:], position=p[:3], color=(255, 0, 200),  # scale below keeps them out of the way
            image=load_jpg(rollout / f"camera_{i}/rgb/0000.jpg", frustum_image_width), format="jpeg",
        ))

    for t in range(n):
        robot.update_cfg(joints[t][idx])
        if t == 0:
            # viser skips setters whose value is unchanged, so frame 0 would carry no cube pose and a looping
            # player would keep the cube at its final (placed) pose through the reach. Force an update:
            # -q is the same rotation, the 0.1 mm offset is immediately overwritten.
            cube.wxyz, cube.position = -obj[0, :4], obj[0, 4:] + 1e-4
        cube.wxyz, cube.position = obj[t, :4], obj[t, 4:]
        for i, fr in enumerate(frustums):
            fr.image = load_jpg(rollout / f"camera_{i}/rgb/{t:04d}.jpg", frustum_image_width)
        rec.insert_sleep(1.0 / fps)
    recording = "ur5e_cube_replay.viser"
    (output_dir / recording).write_bytes(rec.serialize())
    server.stop()

    # Viser web client (same version as the recording) next to the page.
    client = Path(viser.__file__).parent / "client/build"
    shutil.copytree(client, output_dir / "viser-client", dirs_exist_ok=True)

    # Left panel: the human demo from the RealSense front camera.
    frames = output_dir / "_demo_frames"
    frames.mkdir(exist_ok=True)
    for k, step in enumerate(range(demo_steps[0], demo_steps[1] + 1)):
        shutil.copy(demo_rgbd_dir / f"{step:06d}/realsense_front_color.jpg", frames / f"{k:04d}.jpg")
    subprocess.run(["/usr/bin/ffmpeg", "-v", "error", "-y", "-framerate", "10", "-i", str(frames / "%04d.jpg"),
                    "-vf", "scale=960:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_dir / "demo.mp4")], check=True)
    shutil.rmtree(frames)

    (output_dir / "index.html").write_text(PAGE.format(
        recording=recording, cam_pos="0.42,-1.32,0.42", cam_look="0.08,-0.7,0.06"))
    size = (output_dir / recording).stat().st_size / 1e6
    print(f"wrote {output_dir}/index.html and {recording} ({size:.1f} MB)")


if __name__ == "__main__":
    tyro.cli(main)
