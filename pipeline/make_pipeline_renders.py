"""Frame sequences for the pipeline video (make_pipeline_video.py), all from the phone scan.

  colmap   COLMAP's sparse points and the 319 phone camera poses, plus the two RealSense cameras
           (hand-eye calibrated), in the robot base frame, on an orbit over the bench
  splats   one orbit around the cube through the model stack, the same camera for every layer:
           scene.jpg   the splatfacto scene (3D Gaussian splatting, 450,534 Gaussians)
           select.png  the GARField + DiG selection (724 Gaussians: cube + fixture top), RGBA
           cut.png     the metric cut in the robot frame (493 Gaussians: the cube), RGBA
  meshes   sugar.png   SuGaR's surface mesh on the same orbit (RGBA), and
           asset/      a turntable of the cleaned 3,965-triangle sim / ICP asset with its wireframe

The splat states are in nerfstudio model units, the SuGaR mesh and the COLMAP export in the scan
frame (transforms.json); both are put in the robot base frame with the registered similarity
(outputs/table_new/scan_to_base.json) so the orbits can be written in metres.

    PATH=$PWD/.venv/bin:$PATH .venv/bin/python scripts/ur_cube/make_pipeline_renders.py --part colmap
    CUDA_VISIBLE_DEVICES=1 PATH=$PWD/.venv/bin:$PATH .venv/bin/python scripts/ur_cube/make_pipeline_renders.py --part splats
    CUDA_VISIBLE_DEVICES=1 /home/ur10/envs/r2r2r_sugar/bin/python scripts/ur_cube/make_pipeline_renders.py --part meshes
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SCAN = Path("outputs/table_new")
STATE = SCAN / "state_rigid_20261102_190817"
OUT = Path("outputs/ur_cube/media/pipeline_video")
W, H = 960, 600
BG = (19, 21, 23)
BACKDROP = (62, 66, 70)  # make_pipeline_video.py shows isolated objects on this
ACCENT = (95, 181, 192)
WARM = (236, 160, 80)


def look_at(eye, target, up=(0, 0, 1.0)):
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    V = np.eye(4)
    V[:3, :3] = np.stack([r, -u, f])  # OpenCV camera: x right, y down, z forward
    V[:3, 3] = -V[:3, :3] @ eye
    return V


def T_base_model():
    return np.array(json.loads((SCAN / "scan_to_base.json").read_text())["T_base_model"])


def T_base_scan():
    """Scan frame (COLMAP's, as in cameras.json and the SuGaR mesh) -> base; model = s * (D @ [x; 1])."""
    dp = json.loads((SCAN / "splatfacto/2026-11-01_205333/dataparser_transforms.json").read_text())
    S = np.eye(4)
    S[:3, :] = dp["scale"] * np.array(dp["transform"])
    return T_base_model() @ S


def view_into(V, T):
    """World->camera for points in T's source frame, with T's scale divided out (projection-invariant)."""
    s = np.cbrt(np.linalg.det(T[:3, :3]))
    M = np.eye(4)
    M[:3, :3] = V[:3, :3] @ (T[:3, :3] / s)
    M[:3, 3] = (V[:3, :3] @ T[:3, 3] + V[:3, 3]) / s
    return M, s


def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def stack_views(n: int, centre):
    """Orbit around the cube: wide on the fixture first, closing in on the cube."""
    views = []
    for i in range(n):
        k = i / (n - 1)
        az = np.deg2rad(-150 + 120 * k)
        el = np.deg2rad(38 - 10 * ease(k))
        r = 0.42 - 0.20 * ease(k / 0.8)
        eye = centre + r * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        views.append(look_at(eye, centre + [0, 0, -0.004]))
    return views


def intrinsics(f):
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])


# ----------------------------------------------------------------------------- colmap
def part_colmap(n: int) -> None:
    out = OUT / "colmap"
    out.mkdir(parents=True, exist_ok=True)
    tj = json.loads(Path("scans/table_new/transforms.json").read_text())
    # transforms.json and sparse_pc.ply carry nerfstudio's applied_transform on top of COLMAP's frame;
    # the dataparser transform (hence T_base_scan) starts from COLMAP's frame, so undo it first
    A = np.eye(4)
    A[:3, :] = np.array(tj["applied_transform"])
    T = T_base_scan() @ np.linalg.inv(A)
    s = np.cbrt(np.linalg.det(T[:3, :3]))
    ply = Path("scans/table_new/sparse_pc.ply")
    lines = ply.read_text().splitlines()
    data = np.array([list(map(float, ln.split())) for ln in lines[lines.index("end_header") + 1:] if ln.strip()])
    P = data[:, :3] @ T[:3, :3].T + T[:3, 3]
    C = data[:, 3:6].astype(np.uint8)
    keep = (np.abs(P[:, 0] - 0.1) < 1.6) & (np.abs(P[:, 1] + 0.6) < 1.6) & (P[:, 2] > -0.9) & (P[:, 2] < 1.4)
    P, C = P[keep], C[keep]
    fx, w, h = tj["fl_x"], tj["w"], tj["h"]
    phone = []
    for fr in tj["frames"]:
        c2w = np.array(fr["transform_matrix"])  # OpenGL camera: x right, y up, z back
        R = (T[:3, :3] / s) @ c2w[:3, :3]
        phone.append((T[:3, :3] @ c2w[:3, 3] + T[:3, 3], R @ np.diag([1, -1, -1])))  # -> OpenCV axes
    rs = []
    for cam in json.loads(Path("outputs/ur_tape/realsense_calib.json").read_text())["cameras"].values():
        M = np.array(cam["T_cam2base"])
        rs.append((M[:3, 3], M[:3, :3], cam["intrinsics"]))
    print(f"{len(P)} sparse points, {len(phone)} phone cameras; phone centres span "
          f"{np.ptp(np.array([c for c, _ in phone]), 0).round(2)} m")

    ss = 2  # supersample
    Wb, Hb = W * ss, H * ss
    centre = np.array([0.12, -0.62, 0.05])
    for i in range(n):
        k = i / (n - 1)
        az = np.deg2rad(-35 + 80 * k)
        el = np.deg2rad(42 - 6 * k)
        r = 2.75 - 0.45 * ease(k)
        eye = centre + r * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        V = look_at(eye, centre)
        K = intrinsics(780.0) * [[ss], [ss], [1]]
        img = np.zeros((Hb, Wb, 3), np.uint8)
        img[:] = BG
        pc = P @ V[:3, :3].T + V[:3, 3]
        ok = pc[:, 2] > 0.05
        uv = (pc[ok, :2] / pc[ok, 2:3]) @ K[:2, :2].T + K[:2, 2]
        order = np.argsort(-pc[ok, 2])
        uv, col = uv[order].astype(int), C[ok][order]
        for dx in (0, 1, 2):
            for dy in (0, 1, 2):
                u, v = uv[:, 0] + dx, uv[:, 1] + dy
                m = (u >= 0) & (u < Wb) & (v >= 0) & (v < Hb)
                img[v[m], u[m]] = col[m]
        pil = Image.fromarray(img).convert("RGBA")
        layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)

        def proj(X):
            c = V[:3, :3] @ X + V[:3, 3]
            return None if c[2] < 0.05 else tuple((K[:2, :2] @ (c[:2] / c[2]) + K[:2, 2]).tolist())

        def frustum(ctr, R, fx_, w_, h_, depth, colour, width):
            xs, ys = w_ / 2 / fx_ * depth, h_ / 2 / fx_ * depth
            corners = [ctr + R @ np.array([sx * xs, sy * ys, depth]) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            a, cs = proj(ctr), [proj(c) for c in corners]
            if a is None or any(c is None for c in cs):
                return
            for c in cs:
                d.line([a, c], fill=colour, width=width)
            d.line(cs + [cs[0]], fill=colour, width=width)

        for ctr, R in phone:
            frustum(ctr, R, fx, w, h, 0.06, (*ACCENT, 210), 2)
        for ctr, R, intr in rs:
            frustum(ctr, R, intr["fx"], intr["width"], intr["height"], 0.16, (*WARM, 255), 4)
        o = proj(np.zeros(3))
        for axis, colour in ((np.array([0.18, 0, 0]), (226, 84, 74, 255)), (np.array([0, 0.18, 0]), (110, 200, 110, 255)),
                             (np.array([0, 0, 0.18]), (90, 140, 240, 255))):
            e = proj(axis)
            if o and e:
                d.line([o, e], fill=colour, width=5)
        pil = Image.alpha_composite(pil, layer).convert("RGB").resize((W, H), Image.LANCZOS)
        pil.save(out / f"{i:04d}.jpg", quality=90)
    print(f"wrote {n} frames to {out}")


# ----------------------------------------------------------------------------- splats
def part_splats(n: int) -> None:
    import torch
    from gsplat import rasterization

    dev = "cuda"
    load = lambda f: torch.load(f, map_location="cpu", weights_only=False)  # noqa: E731

    def gauss(p):
        p = {k: v.detach().float().to(dev) for k, v in p.items()}
        return dict(means=p["means"], quats=torch.nn.functional.normalize(p["quats"], dim=-1),
                    scales=torch.exp(p["scales"]), opac=torch.sigmoid(p["opacities"]).squeeze(-1),
                    sh=torch.cat([p["features_dc"][:, None], p["features_rest"]], 1))

    sel = gauss(load(SCAN / "_unfiltered_states/state_rigid_20261102_185432.pt"))
    cut = gauss(load(SCAN / "state_rigid_20261102_190817.pt"))
    ck = load(SCAN / "splatfacto/2026-11-01_205333/nerfstudio_models/step-000029999.ckpt")["pipeline"]
    scene = gauss({k.split(".")[-1]: v for k, v in ck.items() if k.startswith("_model.gauss_params.")})
    Tm = T_base_model()
    cm = cut["means"].cpu().numpy() @ Tm[:3, :3].T + Tm[:3, 3]
    centre = (cm.min(0) + cm.max(0)) / 2
    print(f"cube centre (base) {centre.round(4)}, extent {(np.ptp(cm, 0) * 1000).round(1)} mm; "
          f"{len(scene['means'])} / {len(sel['means'])} / {len(cut['means'])} Gaussians")
    views = stack_views(n, centre)
    K = intrinsics(1250.0)
    for sub in ("scene", "select", "cut"):
        (OUT / "stack" / sub).mkdir(parents=True, exist_ok=True)
    (OUT / "stack/views.json").write_text(json.dumps({"W": W, "H": H, "K": K.tolist(), "centre": centre.tolist(),
                                                      "views": [V.tolist() for V in views]}))
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)[None]  # noqa: E731
    for i, V in enumerate(views):
        Vm, s = view_into(V, Tm)
        for name, G in (("scene", scene), ("select", sel), ("cut", cut)):
            img, alpha, _ = rasterization(G["means"], G["quats"], G["scales"], G["opac"], G["sh"], t(Vm), t(K), W, H,
                                          sh_degree=3, near_plane=0.02 / s, far_plane=50.0 / s,
                                          backgrounds=None if name != "scene" else t(np.array(BG) / 255))
            rgb = img[0].clamp(0, 1).cpu().numpy()
            if name == "scene":
                Image.fromarray((rgb * 255).astype(np.uint8)).save(OUT / f"stack/scene/{i:04d}.jpg", quality=90)
            else:
                a = alpha[0, ..., 0].clamp(0, 1).cpu().numpy()
                col = np.where(a[..., None] > 1e-4, rgb / np.maximum(a[..., None], 1e-4), 0).clip(0, 1)
                rgba = np.dstack([col, a])
                Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").save(OUT / f"stack/{name}/{i:04d}.png")
    print(f"wrote {n} x 3 frames to {OUT / 'stack'}")


# ----------------------------------------------------------------------------- meshes
def part_meshes(asset: Path, n_turn: int) -> None:
    import torch
    import trimesh
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import cameras_from_opencv_projection

    dev = "cuda"

    def raster(mesh, W2C, K):
        cams = cameras_from_opencv_projection(torch.tensor(W2C[None, :3, :3], dtype=torch.float32),
                                              torch.tensor(W2C[None, :3, 3], dtype=torch.float32),
                                              torch.tensor(K[None], dtype=torch.float32),
                                              torch.tensor([[H, W]])).to(dev)
        rs = RasterizationSettings(image_size=(H, W), faces_per_pixel=1, bin_size=None, max_faces_per_bin=200000)
        return MeshRasterizer(cams, rs)(mesh)

    def shade(n, W2C, colour):
        n = n @ W2C[:3, :3].T
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        light = np.array([-0.35, -0.55, -0.76])
        light /= np.linalg.norm(light)
        return (0.25 + 0.75 * np.abs(n @ light))[:, None] * np.array(colour)

    # SuGaR mesh on the stack orbit
    vj = json.loads((OUT / "stack/views.json").read_text())
    K = np.array(vj["K"])
    sugar = trimesh.load(next(STATE.glob("sugarfine_*.obj")), force="mesh", process=False)
    sugar.merge_vertices(merge_tex=True, merge_norm=True)
    v, f = np.asarray(sugar.vertices), np.asarray(sugar.faces)
    mesh = Meshes(torch.tensor(v, dtype=torch.float32, device=dev)[None], torch.tensor(f, device=dev)[None])
    vn_all = mesh.verts_normals_packed().cpu().numpy()
    Tb = T_base_scan()
    (OUT / "stack/sugar").mkdir(parents=True, exist_ok=True)
    for i, V in enumerate(vj["views"]):
        W2C, _ = view_into(np.array(V), Tb)
        frag = raster(mesh, W2C, K)
        p2f = frag.pix_to_face[0, ..., 0].cpu().numpy()
        hit = p2f >= 0
        bary = frag.bary_coords[0, ..., 0, :].cpu().numpy()[hit]
        nrm = (bary[..., None] * vn_all[f[p2f[hit]]]).sum(1)
        rgba = np.zeros((H, W, 4))
        rgba[hit, :3] = shade(nrm, W2C, [0.72, 0.74, 0.78])
        rgba[hit, 3] = 1
        Image.fromarray((rgba.clip(0, 1) * 255).astype(np.uint8), "RGBA").save(OUT / f"stack/sugar/{i:04d}.png")
    print(f"SuGaR: {len(f)} triangles, {len(vj['views'])} frames")

    # asset turntable with wireframe
    a = trimesh.load(asset, force="mesh", process=False)
    v, f = np.asarray(a.vertices), np.asarray(a.faces)
    lo, hi = v.min(0), v.max(0)
    ctr = (lo + hi) / 2
    mesh = Meshes(torch.tensor(v, dtype=torch.float32, device=dev)[None], torch.tensor(f, device=dev)[None])
    fn = mesh.faces_normals_packed().cpu().numpy()
    Ka = np.array([[H * 3.4, 0, W / 2], [0, H * 3.4, H / 2], [0, 0, 1.0]])
    out = OUT / "asset"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n_turn):
        az = np.deg2rad(-50 + 360 * i / n_turn)
        el = np.deg2rad(28)
        eye = ctr + 0.30 * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        W2C = look_at(eye, ctr)
        frag = raster(mesh, W2C, Ka)
        p2f = frag.pix_to_face[0, ..., 0].cpu().numpy()
        hit = p2f >= 0
        img = np.zeros((H, W, 3))
        img[:] = np.array(BACKDROP) / 255
        img[hit] = shade(fn[p2f[hit]], W2C, [0.52, 0.55, 0.60])
        pil = Image.fromarray((img.clip(0, 1) * 255).astype(np.uint8)).convert("RGBA")
        seen = np.unique(p2f[hit])
        cam = v @ W2C[:3, :3].T + W2C[:3, 3]
        uv = cam[:, :2] / cam[:, 2:3] @ Ka[:2, :2].T + Ka[:2, 2]
        layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for tri in f[seen]:
            d.line([tuple(uv[j]) for j in (*tri, tri[0])], fill=(*ACCENT, 120), width=1)
        Image.alpha_composite(pil, layer).convert("RGB").save(out / f"{i:04d}.jpg", quality=90)
    print(f"asset: {len(f)} triangles, extent {np.round((hi - lo) * 1000, 1)} mm, {n_turn} frames")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["colmap", "splats", "meshes"], required=True)
    ap.add_argument("--frames", type=int, default=None, help="orbit length (colmap 150, stack 420, asset 150)")
    ap.add_argument("--asset", type=Path, default=Path("data/ur_cube/cube_mesh.obj"))
    a = ap.parse_args()
    if a.part == "colmap":
        part_colmap(a.frames or 150)
    elif a.part == "splats":
        part_splats(a.frames or 420)
    else:
        part_meshes(a.asset, a.frames or 150)


if __name__ == "__main__":
    main()
