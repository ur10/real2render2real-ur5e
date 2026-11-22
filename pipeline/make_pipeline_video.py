"""The project page's pipeline video: one human demo becomes robot training data, stage by stage,
with the model and the numbers at each stage, for all three tasks (cube, tape roll, nut).

Inputs: the frame sequences from make_pipeline_renders.py (COLMAP cameras, the model-stack orbit,
the asset turntable), a label-free splat orbit (make_splat_orbit.py --no-labels), and the existing
media (triptychs, randomized grids, repair comparison, policy grid, tape / nut fit overlays).

Two layouts with the same timeline: `wide` (1280x720, desktop) and `tall` (720x1080, phones); plus
`teaser`, a 26 s text-light loop of the same shots for the homepage tile.

    .venv/bin/python scripts/ur_cube/make_pipeline_video.py --layout wide
    .venv/bin/python scripts/ur_cube/make_pipeline_video.py --layout tall
    .venv/bin/python scripts/ur_cube/make_pipeline_video.py --layout teaser   # 640x360 homepage loop
    .venv/bin/python scripts/ur_cube/make_pipeline_video.py --layout wide --stills 4 30 60   # check frames
"""

import argparse
import json
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS = 30
XFADE = 0.45
REPO = Path(__file__).resolve().parents[2]
PV = REPO / "outputs/ur_cube/media/pipeline_video"
FONTS = REPO / "outputs/ur_cube/fonts"
BG = (19, 21, 23)
PANEL = (28, 31, 34)
INK, MUTED, DIM = (236, 234, 230), (160, 167, 174), (78, 84, 90)
ACCENT, WARM, GOOD = (95, 181, 192), (236, 160, 80), (130, 200, 140)
STAGES = ["SCAN", "SEGMENT", "MESH", "TRACK", "PLAN", "GENERATE", "REPAIR", "TRAIN"]

LAYOUTS = {
    "wide": dict(W=1280, H=720, x=40, step_y=24, step_px=18, title_y=48, title_px=38, title_w=1200,
                 media=(40, 112, 780, 488), bul=(848, 120, 392), head_px=16, body_px=23, gap=17,
                 rail=(78, 1202, 668), rail_px=14, tag_px=18),
    "tall": dict(W=720, H=1080, x=24, step_y=24, step_px=18, title_y=48, title_px=34, title_w=672,
                 media=(24, 152, 672, 420), bul=(24, 596, 672), head_px=17, body_px=26, gap=15,
                 rail=(44, 676, 1024), rail_px=13, tag_px=20),
}


@lru_cache(maxsize=None)
def font(size: int, weight: int = 400, mono: bool = False):
    f = ImageFont.truetype(str(FONTS / ("IBMPlexMono-Medium.ttf" if mono else "SourceSans3.ttf")), size)
    if not mono:
        try:
            f.set_variation_by_axes([weight])
        except Exception:
            pass
    return f


def wrap(draw, text, f, width):
    lines, cur = [], ""
    for w in text.split(" "):
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=f) <= width or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + ([cur] if cur else [])


def ease(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


# ----------------------------------------------------------------------------- media sources
def fit(img, w, h, mode="contain", fill=PANEL):
    ih, iw = img.shape[:2]
    if mode == "cover":
        s = max(w / iw, h / ih)
        nw, nh = int(round(iw * s)), int(round(ih * s))
        r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        x0, y0 = (nw - w) // 2, (nh - h) // 2
        return r[y0:y0 + h, x0:x0 + w]
    s = min(w / iw, h / ih)
    nw, nh = int(round(iw * s)), int(round(ih * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    out = np.empty((h, w, img.shape[2]), img.dtype)
    out[:] = fill if img.shape[2] == 3 else (*fill, 0)
    out[(h - nh) // 2:(h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw] = r
    return out


@lru_cache(maxsize=None)
def clip_frames(path: str, crop: tuple | None, w: int, h: int, mode: str):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if crop:
            x, y, cw, ch = crop
            im = im[y:y + ch, x:x + cw]
        frames.append(fit(im, w, h, mode))
    assert frames, path
    return frames


@lru_cache(maxsize=4096)
def seq_frame(path: str, w: int, h: int, mode: str = "contain"):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA if im.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    return fit(im, w, h, mode)


@lru_cache(maxsize=None)
def still(path: str, w: int, h: int, mode: str = "contain"):
    im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return fit(im, w, h, mode)


@dataclass
class Clip:
    path: str
    crop: tuple | None = None
    speed: float = 1.0
    start: float = 0.0
    mode: str = "contain"
    loop: bool = True

    def frame(self, t, w, h):
        fr = clip_frames(str(REPO / self.path), self.crop, w, h, self.mode)
        i = int((self.start + t * self.speed) * FPS)
        i = i % len(fr) if self.loop else min(i, len(fr) - 1)
        return fr[i]


TRIP = {"real": (12, 62, 624, 350), "track": (648, 62, 624, 350), "render": (1284, 62, 624, 350)}


def trip(task, panel, **kw):
    p = {"cube": "outputs/ur_cube/media/triptych.mp4", "tape": "outputs/ur_tape/media/tape_triptych.mp4",
         "nut": "outputs/ur_nut/media/nut_triptych.mp4"}[task]
    return Clip(p, TRIP[panel], **kw)


def blend(a, b, k):
    if k <= 0:
        return a
    if k >= 1:
        return b
    return (a.astype(np.float32) * (1 - k) + b.astype(np.float32) * k).astype(np.uint8)


def over(base, rgba, k=1.0):
    a = rgba[..., 3:4].astype(np.float32) / 255 * k
    return (base.astype(np.float32) * (1 - a) + rgba[..., :3].astype(np.float32) * a).astype(np.uint8)


BACKDROP = (62, 66, 70)


def outlined(base, rgba, line=True):
    """Overlay a layer with an accent outline and soft glow, so a black object reads on any backdrop."""
    m = (rgba[..., 3] > 110).astype(np.uint8)
    out = base.astype(np.float32)
    if line:
        glow = cv2.GaussianBlur(cv2.dilate(m, np.ones((9, 9), np.uint8)).astype(np.float32), (0, 0), 7)[..., None]
        out = out * (1 - 0.35 * glow) + np.array(ACCENT, np.float32) * 0.35 * glow
    out = over(out.astype(np.uint8), rgba)
    if line:
        edge = (cv2.dilate(m, np.ones((5, 5), np.uint8)) - m).astype(bool)
        out[edge] = ACCENT
    return out


def sequence(items):
    """items: (t0, t1, fn(t_local, w, h)); cross-fades over XFADE where neighbours overlap."""
    def fn(t, w, h):
        cur = [(t0, t1, f) for t0, t1, f in items if t0 - XFADE / 2 <= t < t1 + XFADE / 2]
        if not cur:
            t0, t1, f = items[-1] if t >= items[-1][1] else items[0]
            return f(t - t0, w, h)
        if len(cur) == 1:
            t0, _, f = cur[0]
            return f(max(t - t0, 0), w, h)
        (a0, a1, fa), (b0, _, fb) = cur[0], cur[1]
        k = ease((t - (a1 - XFADE / 2)) / XFADE)
        return blend(fa(t - a0, w, h), fb(max(t - b0, 0), w, h), k)
    return fn


def clipfn(c: Clip):
    return lambda t, w, h: c.frame(t, w, h)


# ----------------------------------------------------------------------------- the model-stack orbit
def stack_media(i0: int, phases, mode: str = "contain"):
    """Orbit frames from i0 at 30 fps; phases: (t, state) with state in scene / select / cut / alone /
    sugar / asset; consecutive states cross-fade over 0.6 s."""
    def layer(state, i, w, h):
        sc = seq_frame(str(PV / f"stack/scene/{i:04d}.jpg"), w, h, mode)
        if state == "scene":
            return sc
        if state in ("select", "cut"):
            grey = sc.astype(np.float32).mean(2, keepdims=True)
            dim = (0.30 * (0.6 * grey + 0.4 * sc) + 0.70 * np.array(BACKDROP) * 0.6).astype(np.uint8)
            return outlined(dim, seq_frame(str(PV / f"stack/{state}/{i:04d}.png"), w, h, mode))
        base = np.empty_like(sc)
        base[:] = BACKDROP
        name = "cut" if state == "alone" else "sugar"
        return outlined(base, seq_frame(str(PV / f"stack/{name}/{i:04d}.png"), w, h, mode), line=state == "alone")

    def fn(t, w, h):
        i = int(np.clip(i0 + t * FPS, 0, 419))
        k = max(j for j, (t0, _) in enumerate(phases) if t0 <= t) if t >= phases[0][0] else 0
        cur = layer(phases[k][1], i, w, h)
        if k + 1 < len(phases) and t > phases[k + 1][0] - 0.6:
            return blend(cur, layer(phases[k + 1][1], i, w, h), ease((t - (phases[k + 1][0] - 0.6)) / 0.6))
        return cur
    return fn


def imgseq(dirname, n, offset=0, mode="contain"):
    return lambda t, w, h: seq_frame(str(PV / dirname / f"{(offset + int(t * FPS)) % n:04d}.jpg"), w, h, mode)


def ring_models():
    """The tape / nut measured-shape overlays: two tape fits stacked on the left, the mug tree and the
    nut on its bolt stand as tall tiles (crops of the front view of each overlay)."""
    tiles_ = [("outputs/ur_tape/dims_overlay_1_0.jpg", (0, 20, 300, 260), "Tape fit, at rest"),
              ("outputs/ur_tape/dims_overlay_1_470.jpg", (0, 20, 300, 260), "Tape fit, on the tree"),
              ("outputs/ur_tape/mug_tree_overlay.jpg", (490, 90, 400, 500), "Mug tree"),
              ("outputs/ur_nut/stand_overlay.jpg", (40, 60, 200, 330), "Nut on the bolt stand")]

    @lru_cache(maxsize=4)
    def build(w, h):
        g = 8
        c1 = int(w * 0.42)
        c2 = (w - c1 - 2 * g) // 2
        boxes = [(0, 0, c1, (h - g) // 2), (0, (h - g) // 2 + g, c1, h - (h - g) // 2 - g),
                 (c1 + g, 0, c2, h), (c1 + 2 * g + c2, 0, w - c1 - 2 * g - c2, h)]
        canvas = Image.new("RGB", (w, h), BG)
        d = ImageDraw.Draw(canvas)
        f = font(max(15, h // 26), 600)
        for (p, (x, y, cw0, ch0), lab), (x0, y0, bw, bh) in zip(tiles_, boxes):
            im = cv2.cvtColor(cv2.imread(str(REPO / p)), cv2.COLOR_BGR2RGB)[y:y + ch0, x:x + cw0]
            canvas.paste(Image.fromarray(fit(im, bw, bh, "cover")), (x0, y0))
            tw = d.textlength(lab, font=f)
            d.rounded_rectangle((x0 + 8, y0 + 8, x0 + 22 + tw, y0 + 18 + f.size), 5, fill=BG)
            d.text((x0 + 15, y0 + 10), lab, font=f, fill=INK)
        return np.asarray(canvas)

    def fn(t, w, h):
        img = build(w, h)
        s = 1 + 0.04 * ease(t / 7)  # slow push-in
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return fn


def tiles(fns_labels, cols):
    """Several live sources side by side (intro)."""
    def fn(t, w, h):
        g = 12
        rows = (len(fns_labels) + cols - 1) // cols
        cw, ch = (w - g * (cols - 1)) // cols, (h - g * (rows - 1)) // rows
        canvas = np.empty((h, w, 3), np.uint8)
        canvas[:] = BG
        for k, (f, _) in enumerate(fns_labels):
            x0, y0 = (k % cols) * (cw + g), (k // cols) * (ch + g)
            canvas[y0:y0 + ch, x0:x0 + cw] = f(t, cw, ch)
        return canvas
    return fn


# ----------------------------------------------------------------------------- scenes
@dataclass
class Scene:
    dur: float
    stage: int | None
    step: str
    title: str
    media: object = None
    tags: list = field(default_factory=list)      # (t, text)
    bullets: list = field(default_factory=list)   # (t, head, body, kind)
    kind: str = "main"                            # main / intro / outro


def scenes():
    cube_stack_a = stack_media(0, [(0, "scene"), (2.6, "select"), (5.6, "cut")])
    cube_stack_b = stack_media(9.0 * FPS - XFADE * FPS, [(0, "cut"), (0.9, "alone"), (2.0, "sugar")])
    asset = imgseq("asset", 150)
    ring_grid = ring_models()
    return [
        Scene(5.2, None, "REAL2RENDER2REAL ON A UR5e", "One human video  →  hundreds of robot demos", kind="intro",
              media=tiles([(clipfn(trip("cube", "real", mode="cover")), "Cube between fixtures · RGB-D"),
                           (clipfn(trip("tape", "real", mode="cover")), "Tape roll onto a mug tree · RGB"),
                           (clipfn(trip("nut", "real", mode="cover")), "Nut onto a bolt · RGB")], cols=3),
              bullets=[(0.6, "INPUT", "A phone video of the bench, and one human demo per task from two calibrated "
                                      "RealSense cameras", ""),
                       (1.4, "OUTPUT", "Rendered robot demonstrations (two camera views + joint targets) to train a "
                                       "visual policy", "")]),
        Scene(9.4, 0, "01 · SCAN", "Phone video → metric 3D twin of the bench",
              media=sequence([(0, 5.0, imgseq("colmap", 150)),
                              (5.0, 9.4, clipfn(Clip("outputs/ur_cube/media/pipeline_video/splat_orbit.mp4", speed=1.9,
                                                     start=0.4, mode="cover", loop=False)))]),
              tags=[(0, "COLMAP: 319 posed phone frames · RealSense cameras in orange · robot base axes"),
                    (5.0, "3D Gaussian splat in the robot frame, ending on a wipe to a real RealSense frame")],
              bullets=[(0.5, "COLMAP  ·  STRUCTURE FROM MOTION", "319 phone frames → camera poses and sparse points", ""),
                       (1.7, "3D GAUSSIAN SPLATTING  ·  SPLATFACTO", "30k steps → 450,534 Gaussians of the bench", ""),
                       (3.0, "METRIC REGISTRATION", "RealSense frames registered into COLMAP, lifted with depth, "
                                                    "RANSAC + Umeyama into the robot base frame", ""),
                       (4.6, "CHECK", "two independent scale estimates agree to 0.2%; 4 mm median residual", "good")]),
        Scene(9.0, 1, "02 · SEGMENT  ·  CUBE", "Learned 3D segmentation picks out the object",
              media=cube_stack_a,
              tags=[(0, "3D Gaussian splat · 450,534 Gaussians"),
                    (2.6, "GARField + DiG selection · 724 Gaussians: cube + fixture top"),
                    (5.6, "Metric cut in the robot frame · 493 Gaussians: the cube")],
              bullets=[(0.5, "GARFIELD  ·  SAM ViT-H", "multi-view SAM masks → scale-conditioned 3D grouping field", ""),
                       (1.7, "DiG  ·  DINOv2 ViT-B/14", "64-d features distilled onto every Gaussian → click to select", ""),
                       (3.1, "WHERE IT BROKE", "the 4 cm black cube stays merged with the fixture under it", "fail"),
                       (5.4, "FIX  ·  METRIC GEOMETRY", "keep Gaussians above the fixture top, near the cube's axis, "
                                                       "in the robot frame", "good")]),
        Scene(8.6, 2, "03 · MESH  ·  CUBE", "SuGaR turns Gaussians into a simulation asset",
              media=sequence([(0, 5.0, cube_stack_b), (5.0, 8.6, asset)]),
              tags=[(0, "The cut Gaussians · 493"), (2.0, "SuGaR surface mesh · 285,568 triangles"),
                    (5.0, "Sim + ICP asset · 3,965 triangles · 46.6 × 43.6 × 41.0 mm")],
              bullets=[(0.5, "SuGaR", "SDF-regularized Gaussians → Poisson surface → refined mesh", ""),
                       (2.0, "ASSET", "cleaned to 3,965 triangles, in metres, in the cube's frame", ""),
                       (3.4, "USED BY", "the ICP tracker (action labels) and the IsaacLab USD (rendered images)", ""),
                       (4.8, "NO CAD", "object geometry comes from the same phone video as the background", "good")]),
        Scene(8.0, 2, "03 · OBJECT MODELS  ·  TAPE + NUT", "Colour video only: measured shapes instead",
              media=ring_grid,
              bullets=[(0.5, "WHY NOT SuGaR", "tape, nut and fixtures were never in the phone scan; "
                                              "their demos have no depth", "fail"),
                       (1.9, "ANALYSIS BY SYNTHESIS", "ring models rendered into both views and fitted to colour "
                                                      "silhouettes (1 − IoU)", ""),
                       (3.4, "MEASURED", "tape: 51.4 / 41.3 mm radii, 24 mm wide · nut: 32 mm across, 21 mm hole", ""),
                       (4.8, "FIXTURES", "mug tree and bolt stand triangulated from both calibrated views", "")]),
        Scene(10.0, 3, "04 · TRACK", "Recover the object's 6-DoF motion from the demo",
              media=sequence([(0, 3.6, clipfn(trip("cube", "track", mode="cover"))),
                              (3.6, 6.8, clipfn(trip("tape", "track", mode="cover"))),
                              (6.8, 10.0, clipfn(trip("nut", "track", mode="cover")))]),
              tags=[(0, "Cube · ICP of the SuGaR mesh in two depth views"),
                    (3.6, "Tape roll · two-view silhouette fit"), (6.8, "Nut · silhouette fit, one-view bridge")],
              bullets=[(0.5, "CUBE  ·  RGB-D", "RSRD (DINOv2 feature tracking) slid off the 25 px cube → "
                                                "ICP of the SuGaR mesh on both depth clouds", ""),
                       (3.6, "TAPE + NUT  ·  RGB", "measured shape fitted to both views' silhouettes, frame by frame", ""),
                       (5.6, "OCCLUSION", "nut hidden from one camera for 1.3 s → bridged along the other camera's ray", ""),
                       (7.4, "SYMMETRY", "nearest equivalent pose (24 for the cube), so no fake spins reach the labels",
                        "good")]),
        Scene(7.2, 4, "05 · PLAN", "Object path → robot motion",
              media=sequence([(0, 3.6, clipfn(trip("cube", "render", mode="cover"))),
                              (3.6, 7.2, clipfn(trip("tape", "render", mode="cover")))]),
              tags=[(0, "UR5e + Robotiq 2F-85 replay · IsaacLab"), (3.6, "Tape roll · mug tree turned in the twin")],
              bullets=[(0.5, "GRASPS", "from object symmetry: 4 cube yaws, 12 rim positions for the rings", ""),
                       (1.8, "WHOLE-TRAJECTORY IK  ·  JAX", "approach, grasp, carry, release, retreat for the "
                                                             "calibrated UR5e + Robotiq 2F-85", ""),
                       (3.3, "CHECKS", "joint limits, IK error, sphere-model collisions", ""),
                       (4.6, "TWIN EDITS", "unreachable as recorded: tree turned ~250°, nut start moved 14 cm in",
                        "fail")]),
        Scene(8.4, 5, "06 · GENERATE", "One demo → hundreds of rendered demos",
              media=sequence([(0, 3.4, clipfn(Clip("outputs/ur_cube/media_v2/randomized.mp4"))),
                              (3.4, 6.0, clipfn(Clip("outputs/ur_tape/media/tape_randomized.mp4"))),
                              (6.0, 8.4, clipfn(Clip("outputs/ur_nut/media/nut_randomized.mp4")))]),
              tags=[(0, "Cube · generated rollouts"), (3.4, "Tape roll · generated rollouts"),
                    (6.0, "Nut · generated rollouts")],
              bullets=[(0.5, "WARP", "new object start: the first 60% of the path re-shaped, the placement kept", ""),
                       (1.8, "RENDER  ·  ISAACLAB", "both calibrated camera views; a 360° render of the splat lights "
                                                     "the scene", ""),
                       (3.2, "RANDOMIZE", "lighting, camera pose, table appearance", ""),
                       (4.4, "VALID ROLLOUTS", "cube 448 / 448  ·  tape 25 / 48  ·  nut 48 / 48", "good")]),
        Scene(7.0, 6, "07 · REPAIR", "Edit the scene, repair the demo",
              media=clipfn(Clip("outputs/ur_cube/media_v2/repair_compare.mp4")),
              bullets=[(0.5, "EDIT", "a scanned bottle on the route: the warped demo carries the cube through it", "fail"),
                       (1.9, "TRAJECTORY OPTIMIZATION  ·  JAXLS", "Levenberg–Marquardt; arm, gripper and camera "
                                                                   "bracket as spheres; grasp and placement pinned", ""),
                       (3.6, "RESULT", "21 bottle placements: 5 valid as generated → 17 after repair", "good")]),
        Scene(8.4, 7, "08 · TRAIN  ·  CUBE, IN THE TWIN", "Does the generated data help a policy?",
              media=clipfn(Clip("outputs/ur_cube/media_v2/policy_clean.mp4")),
              bullets=[(0.5, "POLICY", "ResNet-18 on two camera views → 16-step joint-target chunks", ""),
                       (1.8, "CLEAN HELD-OUT SCENES", "3%  →  94%   one demo → 400 generated", "num"),
                       (3.2, "BOTTLE ON THE ROUTE", "4%  →  85%   filtered → repaired demos", "num"),
                       (4.6, "NEXT", "real-robot trials on the same bench", "")]),
        Scene(5.0, None, "THE MODEL STACK", "Phone video in, robot training data out", kind="outro",
              media=clipfn(Clip("outputs/ur_cube/media_v2/randomized.mp4", mode="cover")),
              bullets=[(0.3, "LEARNED MODELS", "COLMAP · 3D Gaussian Splatting · GARField (SAM) · DiG (DINOv2) · SuGaR", ""),
                       (0.9, "GEOMETRY + CONTROL", "RGB-D ICP · silhouette fitting · JAX IK + trajectory optimization · "
                                                   "IsaacLab · ResNet-18 policy", ""),
                       (1.5, "CODE", "github.com/ur10/real2render2real-ur5e", "good")]),
    ]


# ----------------------------------------------------------------------------- drawing
def draw_rail(d, L, active, done_all=False):
    x0, x1, y = L["rail"]
    n = len(STAGES)
    xs = [x0 + (x1 - x0) * k / (n - 1) for k in range(n)]
    d.line((xs[0], y, xs[-1], y), fill=DIM, width=2)
    if done_all:
        d.line((xs[0], y, xs[-1], y), fill=ACCENT, width=3)
    elif active is not None:
        d.line((xs[0], y, xs[active], y), fill=ACCENT, width=3)
    f = font(L["rail_px"], mono=True)
    for k, (x, name) in enumerate(zip(xs, STAGES)):
        if done_all or (active is not None and k < active):
            d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=ACCENT)
            col = MUTED
        elif k == active:
            d.ellipse((x - 10, y - 10, x + 10, y + 10), outline=ACCENT, width=2)
            d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=ACCENT)
            col = ACCENT
        else:
            d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=BG, outline=DIM, width=2)
            col = DIM
        tw = d.textlength(name, font=f)
        d.text((x - tw / 2, y + 15), name, font=f, fill=col)


def fade_col(c, a, bg=BG):
    return tuple(int(bg[i] + (c[i] - bg[i]) * a) for i in range(3))


def draw_bullets(d, L, sc, t, x, y, width, big=False):
    hf = font(L["head_px"], mono=True)
    bf = font(L["body_px"] + (2 if big else 0), 400)
    nf = font(int(L["body_px"] * 1.55), 700)
    for t0, head, body, kind in sc.bullets:
        a = ease((t - t0) / 0.45)
        if a <= 0:
            break
        dy = int((1 - a) * 10)
        hcol = {"fail": WARM, "good": GOOD}.get(kind, ACCENT)
        d.text((x, y + dy), head, font=hf, fill=fade_col(hcol, a))
        y += int(hf.size * 1.45)
        if kind == "num":
            num, rest = body.split("   ", 1)
            d.text((x, y + dy - 2), num, font=nf, fill=fade_col(INK, a))
            y += int(nf.size * 1.12)
            for ln in wrap(d, rest, bf, width):
                d.text((x, y + dy), ln, font=bf, fill=fade_col(MUTED, a))
                y += int(bf.size * 1.25)
        else:
            for ln in wrap(d, body, bf, width):
                d.text((x, y + dy), ln, font=bf, fill=fade_col(INK, a))
                y += int(bf.size * 1.25)
        y += L["gap"]
    return y


def draw_tag(img, d, L, text, a, x, y, maxw):
    if not text or a <= 0:
        return
    f = font(L["tag_px"], 600)
    lines = wrap(d, text, f, maxw - 28)
    lh = int(f.size * 1.25)
    w = max(d.textlength(ln, font=f) for ln in lines) + 24
    h = lh * len(lines) + 14
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    ld.rounded_rectangle((x, y, x + w, y + h), 7, fill=(*BG, int(225 * a)))
    for k, ln in enumerate(lines):
        ld.text((x + 12, y + 6 + k * lh), ln, font=f, fill=(*INK, int(255 * a)))
    img.alpha_composite(layer)


def render_scene(sc: Scene, t: float, L) -> np.ndarray:
    W, H, x = L["W"], L["H"], L["x"]
    img = Image.new("RGBA", (W, H), (*BG, 255))
    d = ImageDraw.Draw(img)
    tall = W < H
    if sc.kind in ("intro", "outro"):
        if sc.kind == "intro":
            if tall:
                th = 178
                mw, mh, my = W - 2 * x, 3 * th + 2 * 12, 204
                med = Image.fromarray(tiles_column(sc.media, t, mw, mh))
            else:
                th = 236
                mw, mh, my = W - 2 * x, th, 250
                med = Image.fromarray(sc.media(t, mw, mh))
            img.paste(med, (x, my))
            labels = ["Cube between fixtures · RGB-D", "Tape roll onto a mug tree · RGB", "Nut onto a bolt · RGB"]
            f = font(L["tag_px"], 600)
            for k, lab in enumerate(labels):
                if tall:
                    tx, ty = x, my + k * (th + 12)
                else:
                    cw = (mw - 24) // 3
                    tx, ty = x + k * (cw + 12), my
                draw_tag(img, d, L, lab, 1.0, tx + 10, ty + 10, 400)
            d = ImageDraw.Draw(img)
        else:
            bgv = sc.media(t, W, H) if not tall else fit(sc.media(t, W, int(W * 830 / 1936)), W, H, "cover")
            img.paste(Image.fromarray((bgv.astype(np.float32) * 0.16 + np.array(BG) * 0.84).astype(np.uint8)), (0, 0))
            d = ImageDraw.Draw(img)
        sf = font(L["step_px"] + 2, mono=True)
        tf = font(int(L["title_px"] * (1.25 if sc.kind == "intro" else 1.15)), 700)
        ty = 46 if sc.kind == "intro" else (150 if not tall else 200)
        d.text((x, ty), sc.step, font=sf, fill=ACCENT)
        yy = ty + int(sf.size * 1.8)
        for ln in wrap(d, sc.title, tf, W - 2 * x):
            d.text((x, yy), ln, font=tf, fill=INK)
            yy += int(tf.size * 1.12)
        if sc.kind == "intro":
            by = my + mh + (28 if not tall else 24)
            if not tall:
                # two bullets side by side under the tiles
                colw = (W - 2 * x - 40) // 2
                for k, b in enumerate(sc.bullets):
                    one = Scene(0, None, "", "", bullets=[b])
                    draw_bullets(d, L, one, t, x + k * (colw + 40), by, colw)
            else:
                draw_bullets(d, L, sc, t, x, by, W - 2 * x)
            draw_rail(d, L, None)
        else:
            draw_bullets(d, L, sc, t, x, yy + 30, W - 2 * x, big=True)
            draw_rail(d, L, None, done_all=True)
        return np.asarray(img.convert("RGB"))

    # main scenes
    d.text((x, L["step_y"]), sc.step, font=font(L["step_px"], mono=True), fill=ACCENT)
    tf = font(L["title_px"], 700)
    yy = L["title_y"]
    for ln in wrap(d, sc.title, tf, L["title_w"]):
        d.text((x, yy), ln, font=tf, fill=INK)
        yy += int(tf.size * 1.12)
    mx, my, mw, mh = L["media"]
    img.paste(Image.fromarray(sc.media(t, mw, mh)), (mx, my))
    if sc.tags:
        k = max(j for j, (t0, _) in enumerate(sc.tags) if t0 <= t) if t >= sc.tags[0][0] else 0
        t0, text = sc.tags[k]
        a = ease((t - t0) / 0.35) if k > 0 else 1.0
        draw_tag(img, d, L, text, a, mx + 12, my + 12, mw - 24)
        d = ImageDraw.Draw(img)
    bx, by, bw = L["bul"]
    draw_bullets(d, L, sc, t, bx, by, bw)
    draw_rail(d, L, sc.stage)
    return np.asarray(img.convert("RGB"))


def tiles_column(fn, t, w, h):
    """The intro's three live tiles stacked for the tall layout (fn lays them out in a row)."""
    g = 12
    th = (h - 2 * g) // 3
    out = np.empty((h, w, 3), np.uint8)
    out[:] = BG
    row = fn(t, 3 * w + 2 * g, th)  # three w x th tiles side by side
    for k in range(3):
        out[k * (th + g):k * (th + g) + th] = row[:, k * (w + g):k * (w + g) + w]
    return out


# ----------------------------------------------------------------------------- homepage preview
def teaser_shots():
    """A short, text-light loop for the 300 px homepage tile: one shot per stage, one big label."""
    orbit = "outputs/ur_cube/media/pipeline_video/splat_orbit.mp4"
    return [
        (2.4, "COLMAP", "319 posed phone frames", imgseq("colmap", 150, 30, "cover")),
        (2.4, "3D Gaussian splat", "the bench, in the robot frame", clipfn(Clip(orbit, speed=1.6, start=0.6, mode="cover"))),
        (2.3, "GARField + DiG", "SAM groups, DINOv2 features", stack_media(70, [(0, "select")], "cover")),
        (2.1, "Metric cut", "the cube: 493 Gaussians", stack_media(140, [(0, "cut")], "cover")),
        (2.3, "SuGaR mesh", "Gaussians to a surface", stack_media(200, [(0, "sugar")], "cover")),
        (2.3, "Measured shapes", "tape + nut, from colour video", ring_models()),
        (2.5, "6-DoF tracking", "two calibrated cameras", clipfn(trip("cube", "track", mode="cover", start=0.6))),
        (2.3, "Robot replay", "UR5e + Robotiq · IsaacLab", clipfn(trip("tape", "render", mode="cover", start=0.8))),
        (2.5, "Generate", "1 demo → 448 rollouts", clipfn(Clip("outputs/ur_cube/media_v2/randomized.mp4", mode="cover"))),
        (2.5, "Repair", "detour around a new obstacle", clipfn(Clip("outputs/ur_cube/media_v2/repair_compare.mp4",
                                                                    mode="cover", start=1.5))),
        (2.7, "Train", "3% → 94% success, in the twin", clipfn(Clip("outputs/ur_cube/media_v2/policy_clean.mp4",
                                                                      mode="cover", start=3.0))),
    ]


def render_teaser(out: Path, W=640, H=360, crf=28):
    shots = teaser_shots()
    starts, t = [], 0.0
    for d, *_ in shots:
        starts.append(t)
        t += d - XFADE
    total = t + XFADE
    grad = np.linspace(0, 1, H)[:, None, None] ** 2.2 * 0.85

    def picture(k, tl):
        img = shots[k][3](tl, W, H).astype(np.float32)
        return (img * (1 - grad) + np.array(BG, np.float32) * grad).astype(np.uint8)

    def label(img, k, a):
        if a <= 0:
            return img
        _, lab, sub, _ = shots[k]
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dr = ImageDraw.Draw(layer)
        dr.text((22, H - 92), lab, font=font(40, 700), fill=(*INK, int(255 * a)))
        dr.text((23, H - 44), sub, font=font(25, 500), fill=(*ACCENT, int(255 * a)))
        dr.text((22, 16), f"{k + 1:02d} / {len(shots)}", font=font(18, mono=True), fill=(*INK, int(255 * a)))
        return np.asarray(Image.alpha_composite(Image.fromarray(img).convert("RGBA"), layer).convert("RGB"))

    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    for i in range(int(total * FPS)):
        T = i / FPS
        cur = [k for k, s0 in enumerate(starts) if s0 <= T < s0 + shots[k][0]]
        if len(cur) == 1:
            fr = label(picture(cur[0], T - starts[cur[0]]), cur[0], 1.0)
        else:
            a, b = cur[0], cur[1]
            k = ease((T - starts[b]) / XFADE)
            pic = blend(picture(a, T - starts[a]), picture(b, T - starts[b]), k)
            fr = label(pic, a, 1 - 2 * k) if k < 0.5 else label(pic, b, 2 * k - 1)
        ff.stdin.write(fr.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"wrote {out} ({total:.1f} s, {out.stat().st_size / 1e6:.1f} MB)")


def timeline(scs):
    starts, t = [], 0.0
    for sc in scs:
        starts.append(t)
        t += sc.dur - XFADE
    return starts, t + XFADE


def frame_at(scs, starts, T, L):
    cur = [k for k, s in enumerate(starts) if s <= T < s + scs[k].dur]
    if len(cur) == 1:
        k = cur[0]
        return render_scene(scs[k], T - starts[k], L)
    if not cur:
        return render_scene(scs[-1], scs[-1].dur, L)
    a, b = cur[0], cur[1]
    A, B = render_scene(scs[a], T - starts[a], L), render_scene(scs[b], T - starts[b], L)
    k = ease((T - starts[b]) / XFADE)
    bg = np.empty_like(A)
    bg[:] = BG
    out = np.array(blend(A, bg, min(1.0, 2 * k)) if k < 0.5 else blend(bg, B, 2 * k - 1))  # text dips through the bg
    if scs[a].kind == scs[b].kind == "main":  # the picture itself cross-fades
        mx, my, mw, mh = L["media"]
        out[my:my + mh, mx:mx + mw] = blend(A[my:my + mh, mx:mx + mw], B[my:my + mh, mx:mx + mw], k)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", choices=list(LAYOUTS) + ["teaser"], default="wide")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--stills", type=float, nargs="*", default=None, help="only write these times (s) as JPGs")
    ap.add_argument("--crf", type=int, default=25)
    a = ap.parse_args()
    if a.layout == "teaser":
        render_teaser(a.output or PV / "pipeline_teaser.mp4")
        return
    L = LAYOUTS[a.layout]
    scs = scenes()
    starts, total = timeline(scs)
    out = a.output or PV / f"pipeline_{a.layout}.mp4"
    chapters = [{"t": round(s, 2), "step": sc.step, "title": sc.title} for s, sc in zip(starts, scs)]
    (PV / "chapters.json").write_text(json.dumps({"duration": round(total, 2), "chapters": chapters}, indent=1))
    print(f"{len(scs)} scenes, {total:.1f} s")
    if a.stills is not None:
        for T in a.stills:
            Image.fromarray(frame_at(scs, starts, T, L)).save(PV / f"still_{a.layout}_{T:05.1f}.jpg", quality=90)
        return
    W, H = L["W"], L["H"]
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf),
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    n = int(total * FPS)
    for i in range(n):
        ff.stdin.write(frame_at(scs, starts, i / FPS, L).tobytes())
        if i % 300 == 0:
            print(f"  frame {i}/{n}", flush=True)
    ff.stdin.close()
    ff.wait()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
