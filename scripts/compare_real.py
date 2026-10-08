#!/usr/bin/env python3
"""So ảnh render camera ngực với ảnh thật từ D435i: ghép cạnh nhau, chồng mờ, và chồng biên cạnh.

Ví dụ:
  python scripts/compare_real.py --real data/real_color.png
  python scripts/compare_real.py --real data/real_color.png --set camera.pitch_deg=43 --set box.gap_from_column=0.15

--set ghi đè số đo trong config tạm thời (không sửa scene.yaml) và dựng scene tạm để render.
Khi đã khớp, chép các giá trị vào config/scene.yaml rồi chạy lại build_scene.py.
"""

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openarm_sim.render import OffscreenRenderer  # noqa: E402  (đặt MUJOCO_GL)

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from openarm_sim import DEFAULT_CONFIG, ROOT  # noqa: E402
from openarm_sim.scene import load_config, load_scene, write_scene  # noqa: E402


def apply_overrides(cfg: dict, sets: list[str]) -> dict:
    cfg = copy.deepcopy(cfg)
    for s in sets:
        key, _, val = s.partition("=")
        node = cfg
        *path, leaf = key.strip().split(".")
        for k in path:
            node = node[k]
        if leaf not in node:
            raise KeyError(f"Không có khoá '{key}' trong config")
        node[leaf] = yaml.safe_load(val)
    return cfg


def edges(rgb: np.ndarray, thresh: float = 25.0) -> np.ndarray:
    g = rgb.astype(np.float32).mean(axis=2)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    return np.hypot(gx, gy) > thresh


def label(img: Image.Image, text: str) -> Image.Image:
    img = img.copy()
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, 8 * len(text) + 12, 22], fill=(0, 0, 0))
    dr.text((6, 5), text, fill=(255, 255, 255))
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True, help="ảnh màu thật từ D435i (png/jpg)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="ghi đè config, vd camera.pitch_deg=43 (lặp được)")
    ap.add_argument("--camera", default="color", choices=["color", "depth"],
                    help="camera sim để so (mặc định color)")
    ap.add_argument("--fovy", type=float, default=None,
                    help="ghi đè fovy (độ) của camera sim, khi ảnh thật dùng độ phân giải/crop khác")
    ap.add_argument("--alpha", type=float, default=0.5, help="độ mờ của ảnh render khi chồng")
    ap.add_argument("--grid", type=int, default=0, help="vẽ lưới N×N để dễ so (0 = tắt)")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "compare"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = apply_overrides(load_config(args.config), args.set)
    if args.fovy is not None:
        cfg["camera"][args.camera]["fovy_deg"] = args.fovy
    scene = out / "_compare_scene.xml"
    write_scene(cfg, scene)
    m, d = load_scene(scene)
    cam = cfg["camera"][args.camera]

    real = Image.open(args.real).convert("RGB")
    w, h = real.size
    cw, ch = cam["resolution"]
    if abs(w / h - cw / ch) > 0.01:
        print(f"CẢNH BÁO: tỉ lệ ảnh thật {w}x{h} khác camera sim {cw}x{ch}. fovy là FOV dọc; "
              f"nếu ảnh thật bị crop, dùng --fovy với FOV dọc thật của ảnh đó.")
    with OffscreenRenderer(m, w, h) as r:
        sim = Image.fromarray(r.rgb(d, cam["name"]))

    real_a, sim_a = np.asarray(real), np.asarray(sim)
    blend = Image.blend(real, sim, args.alpha)
    edge = real_a.copy()
    edge[edges(sim_a)] = (255, 40, 40)            # biên của ảnh sim (đỏ) chồng lên ảnh thật
    edge_img = Image.fromarray(edge)
    imgs = {"side_by_side": None, "blend": blend, "edges": edge_img}
    if args.grid:
        for k in ("blend", "edges"):
            dr = ImageDraw.Draw(imgs[k])
            for i in range(1, args.grid):
                dr.line([(w * i // args.grid, 0), (w * i // args.grid, h)], fill=(0, 255, 0))
                dr.line([(0, h * i // args.grid), (w, h * i // args.grid)], fill=(0, 255, 0))
    sbs = Image.new("RGB", (2 * w, h))
    sbs.paste(label(real, "real"), (0, 0))
    sbs.paste(label(sim, "sim " + cam["name"]), (w, 0))
    imgs["side_by_side"] = sbs
    sim.save(out / "sim.png")
    for k, im in imgs.items():
        im.save(out / f"{k}.png")
    print("Ghi đè:", args.set or "(không)")
    print("Đã ghi:", ", ".join(str(out / f"{k}.png") for k in ["sim", *imgs]))


if __name__ == "__main__":
    main()
