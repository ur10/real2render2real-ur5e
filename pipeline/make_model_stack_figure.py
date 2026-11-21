"""Panels for the "model stack" figure: how the cube goes from a phone video to a simulation asset.

Every panel except the last is drawn from the same phone-scan camera, so the stages line up:
  photo   the phone video frame (input to COLMAP + splatfacto)
  splat   the splatfacto scene (3D Gaussian splatting) rendered from that camera
  select  the GARField + DiG selection (SAM-supervised groups, DINOv2 features): cube + fixture top
  cut     the same selection cut above the fixture top in the robot base frame: the cube only
  sugar   SuGaR's surface mesh extracted from those Gaussians
  asset   the cleaned mesh used by the ICP tracker and the IsaacLab USD (close-up, own camera)

Frames: the DiG states and the splatfacto checkpoint are in nerfstudio model units, SuGaR's mesh
and cameras.json in the scan frame (the one the dataparser transform maps to model units). The
asset mesh is in the cube frame (mesh_to_object_frame.py), so it gets its own close-up camera.

Two parts, one per environment (gsplat lives in the data-gen venv, pytorch3d in the SuGaR env;
Open3D's offscreen renderer returns black frames on this machine):
    CUDA_VISIBLE_DEVICES=1 PATH=$PWD/.venv/bin:$PATH .venv/bin/python \
        scripts/ur_cube/make_model_stack_figure.py --part splats
    CUDA_VISIBLE_DEVICES=1 /home/ur10/envs/r2r2r_sugar/bin/python \
        scripts/ur_cube/make_model_stack_figure.py --part meshes
    .venv/bin/python scripts/ur_cube/make_model_stack_figure.py --part grid   # labelled 3x2 figure
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

SCAN = Path("outputs/table_new")
STATE = SCAN / "state_rigid_20261102_190817"
BG = 0.09


def scan_camera(frame: str, half: int, size: int, centre_uv=None):
    """World->camera (OpenCV) for the scan frame and for model units (times the model scale), and
    the intrinsics of a size x size crop of the phone frame centred on centre_uv."""
    dp = json.loads((SCAN / "dig/2026-11-02_184246/dataparser_transforms.json").read_text())
    s, D = dp["scale"], np.array(dp["transform"])  # model = s * (D @ [x_scan; 1])
    cam = next(c for c in json.loads((STATE / "cameras.json").read_text()) if c["img_name"] == frame)
    R, p = np.array(cam["rotation"]), np.array(cam["position"])  # camera-to-scan rotation, centre
    W2C = np.eye(4)
    W2C[:3, :3], W2C[:3, 3] = R.T, -R.T @ p
    Vm = np.eye(4)  # projection is scale-invariant, so the similarity's scale is multiplied out
    Vm[:3, :3] = W2C[:3, :3] @ D[:3, :3].T
    Vm[:3, 3] = s * (W2C[:3, 3] - W2C[:3, :3] @ D[:3, :3].T @ D[:3, 3])
    cx, cy = centre_uv
    x0 = int(np.clip(cx - half, 0, cam["width"] - 2 * half))
    y0 = int(np.clip(cy - half, 0, cam["height"] - 2 * half))
    k = size / (2 * half)
    K = np.array([[cam["fx"] * k, 0, (cam["cx"] - x0) * k], [0, cam["fy"] * k, (cam["cy"] - y0) * k], [0, 0, 1]])
    return dict(W2C=W2C, Vm=Vm, K=K, box=(x0, y0, x0 + 2 * half, y0 + 2 * half), cam=cam, centre=p)


def part_splats(out: Path, frame: str, half: int, size: int) -> None:
    from gsplat import rasterization

    dev = "cuda"

    def gauss(p):
        p = {k: v.detach().float().to(dev) for k, v in p.items()}
        return dict(means=p["means"], quats=torch.nn.functional.normalize(p["quats"], dim=-1),
                    scales=torch.exp(p["scales"]), opac=torch.sigmoid(p["opacities"]).squeeze(-1),
                    sh=torch.cat([p["features_dc"][:, None], p["features_rest"]], 1))

    load = lambda f: torch.load(f, map_location="cpu", weights_only=False)  # noqa: E731
    sel = gauss(load(SCAN / "_unfiltered_states/state_rigid_20261102_185432.pt"))
    cut = gauss(load(SCAN / "state_rigid_20261102_190817.pt"))
    ck = load(SCAN / "splatfacto/2026-11-01_205333/nerfstudio_models/step-000029999.ckpt")["pipeline"]
    scene = gauss({k.split(".")[-1]: v for k, v in ck.items() if k.startswith("_model.gauss_params.")})

    # centre the crop on the selection (cube + fixture top) as it projects into this camera
    c0 = scan_camera(frame, half, size, (0, 0))
    m = sel["means"].cpu().numpy() @ c0["Vm"][:3, :3].T + c0["Vm"][:3, 3]
    uv = m[:, :2] / m[:, 2:3] * [c0["cam"]["fx"], c0["cam"]["fy"]] + [c0["cam"]["cx"], c0["cam"]["cy"]]
    centre = np.median(uv, 0)
    c = scan_camera(frame, half, size, centre)
    (out / "crop.json").write_text(json.dumps({"frame": frame, "half": half, "size": size, "centre_uv": centre.tolist()}))
    print(f"{frame}: crop {c['box']}; {len(scene['means'])} scene / {len(sel['means'])} selected / "
          f"{len(cut['means'])} cube Gaussians")

    Image.open(f"scans/table_new/images/{frame}.png").convert("RGB").crop(c["box"]).resize(
        (size, size), Image.LANCZOS).save(out / "photo.jpg", quality=88)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)[None]  # noqa: E731
    for name, G in (("splat", scene), ("select", sel), ("cut", cut)):
        img, _, _ = rasterization(G["means"], G["quats"], G["scales"], G["opac"], G["sh"], t(c["Vm"]), t(c["K"]),
                                  size, size, sh_degree=3, near_plane=0.01, far_plane=100.0,
                                  backgrounds=torch.full((1, 3), BG, device=dev))
        Image.fromarray((img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(out / f"{name}.jpg", quality=88)


def part_meshes(out: Path, asset: Path) -> None:
    import trimesh
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import cameras_from_opencv_projection

    dev = "cuda"
    crop = json.loads((out / "crop.json").read_text())
    size = crop["size"]

    def render(v, f, W2C, K, colour, flat=False):
        """Rasterize with pytorch3d, shade two-sided Lambert with a head light (mesh winding varies)."""
        V = torch.tensor(v, dtype=torch.float32, device=dev)[None]
        F = torch.tensor(f, dtype=torch.int64, device=dev)[None]
        mesh = Meshes(V, F)
        cams = cameras_from_opencv_projection(torch.tensor(W2C[None, :3, :3], dtype=torch.float32),
                                              torch.tensor(W2C[None, :3, 3], dtype=torch.float32),
                                              torch.tensor(K[None], dtype=torch.float32),
                                              torch.tensor([[size, size]])).to(dev)
        frag = MeshRasterizer(cams, RasterizationSettings(image_size=size, faces_per_pixel=1, bin_size=0))(mesh)
        p2f = frag.pix_to_face[0, ..., 0].cpu().numpy()
        hit = p2f >= 0
        if flat:
            n = mesh.faces_normals_packed().cpu().numpy()[p2f[hit]]
        else:  # smooth: barycentric blend of vertex normals
            vn = mesh.verts_normals_packed().cpu().numpy()[f[p2f[hit]]]  # (P, 3, 3)
            n = (frag.bary_coords[0, ..., 0, :].cpu().numpy()[hit][..., None] * vn).sum(1)
        n = n @ W2C[:3, :3].T
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        light = np.array([-0.35, -0.55, -0.76])  # towards the camera, from above-left
        light /= np.linalg.norm(light)
        shade = 0.28 + 0.72 * np.abs(n @ light)
        img = np.full((size, size, 3), BG)
        img[hit] = shade[:, None] * np.array(colour)
        return (img.clip(0, 1) * 255).astype(np.uint8), p2f

    c = scan_camera(crop["frame"], crop["half"], size, crop["centre_uv"])
    sugar = trimesh.load(next(STATE.glob("sugarfine_*.obj")), force="mesh", process=False)
    sugar.merge_vertices(merge_tex=True, merge_norm=True)  # the textured export splits every UV seam
    print(f"SuGaR mesh: {len(sugar.faces)} triangles, {len(sugar.vertices)} vertices after merging")
    img, _ = render(np.asarray(sugar.vertices), np.asarray(sugar.faces), c["W2C"], c["K"], [0.70, 0.72, 0.76])
    Image.fromarray(img).save(out / "sugar.jpg", quality=88)

    a = trimesh.load(asset, force="mesh", process=False)
    v, f = np.asarray(a.vertices), np.asarray(a.faces)
    lo, hi = v.min(0), v.max(0)
    print(f"asset: {len(f)} triangles, {len(v)} vertices, extent {np.round((hi - lo) * 1000, 1)} mm")
    ctr = (lo + hi) / 2
    eye = ctr + np.array([0.08, -0.10, 0.075])
    fw = (ctr - eye) / np.linalg.norm(ctr - eye)
    r = np.cross(fw, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, fw)
    W2C = np.eye(4)
    W2C[:3, :3] = np.stack([r, -u, fw])  # OpenCV: x right, y down, z forward
    W2C[:3, 3] = -W2C[:3, :3] @ eye
    K = np.array([[size * 1.8, 0, size / 2], [0, size * 1.8, size / 2], [0, 0, 1]])
    img, p2f = render(v, f, W2C, K, [0.50, 0.52, 0.57], flat=True)
    # wireframe of the faces that are visible somewhere in the image
    seen = np.unique(p2f[p2f >= 0])
    cam_pts = v @ W2C[:3, :3].T + W2C[:3, 3]
    uv = cam_pts[:, :2] / cam_pts[:, 2:3] @ K[:2, :2].T + K[:2, 2]
    pil = Image.fromarray(img).convert("RGBA")
    layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for tri in f[seen]:
        draw.line([tuple(uv[i]) for i in (*tri, tri[0])], fill=(240, 190, 90, 105), width=1)
    Image.alpha_composite(pil, layer).convert("RGB").save(out / "asset.jpg", quality=88)


PANELS = [  # (panel, title, subtitle) for the labelled grid
    ("photo", "1  Phone video frame", "input to COLMAP + splatfacto"),
    ("splat", "2  3D Gaussian splat", "splatfacto: 450,534 Gaussians"),
    ("select", "3  GARField + DiG selection", "SAM groups, DINOv2 features: 724 Gaussians"),
    ("cut", "4  Metric cut, robot frame", "above the fixture top: 493 Gaussians"),
    ("sugar", "5  SuGaR surface mesh", "285,568 triangles"),
    ("asset", "6  Sim / ICP asset", "3,965 triangles, metres, cube frame"),
]


def part_grid(out: Path, n: int = 400, pad: int = 10, cap: int = 58) -> None:
    from PIL import ImageFont

    font = Path("outputs/ur_cube/fonts/SourceSans3.ttf")
    title, sub = ImageFont.truetype(str(font), 22), ImageFont.truetype(str(font), 17)
    W = Image.new("RGB", (3 * n + 4 * pad, 2 * (n + cap) + 3 * pad), (255, 255, 255))
    draw = ImageDraw.Draw(W)
    for i, (name, t, s) in enumerate(PANELS):
        x, y = pad + (i % 3) * (n + pad), pad + (i // 3) * (n + cap + pad)
        W.paste(Image.open(out / f"{name}.jpg").resize((n, n), Image.LANCZOS), (x, y))
        draw.text((x + 2, y + n + 6), t, font=title, fill=(20, 20, 24))
        draw.text((x + 2, y + n + 33), s, font=sub, fill=(95, 99, 108))
    W.save(out / "model_stack_grid.jpg", quality=86)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["splats", "meshes", "grid"], required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/ur_cube/media/model_stack"))
    ap.add_argument("--frame", default="frame_00006")
    ap.add_argument("--half", type=int, default=150, help="crop half-size in phone-frame pixels")
    ap.add_argument("--size", type=int, default=480, help="output panel size")
    ap.add_argument("--asset", type=Path, default=Path("data/ur_cube/cube_mesh.obj"))
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.part == "splats":
        part_splats(a.output_dir, a.frame, a.half, a.size)
    elif a.part == "meshes":
        part_meshes(a.output_dir, a.asset)
    else:
        part_grid(a.output_dir)
    print(f"wrote {sorted(p.name for p in a.output_dir.glob('*.jpg'))}")


if __name__ == "__main__":
    main()
