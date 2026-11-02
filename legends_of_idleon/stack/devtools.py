"""Helpers for building templates from screenshots.

    python -m legends_of_idleon stack snap shot.png
    python -m legends_of_idleon stack crop shot.png play_button 800 600 1100 680
    python -m legends_of_idleon stack find shot.png play_button
    python -m legends_of_idleon stack blobs shot.png 100 150 150 130 255 255

``crop`` and ``find`` work on any OS (so they can run off the game machine);
``snap`` needs Windows.
"""

from __future__ import annotations

import argparse
import json

import cv2

from . import ui


def _load_screen(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read image: {path}")
    return ui.normalize(image)


def cmd_snap(args):
    from ..screen import grab_screen

    image = ui.normalize(grab_screen())
    cv2.imwrite(args.output, image)
    print(f"saved {args.output} ({image.shape[1]}x{image.shape[0]})")


def cmd_crop(args):
    screen = _load_screen(args.screenshot)
    x1, y1, x2, y2 = args.box
    patch = screen[y1:y2, x1:x2]
    if patch.size == 0:
        raise SystemExit("empty crop; check the box coordinates (x1 y1 x2 y2)")
    ui.UI_TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    png = ui.UI_TEMPLATE_DIR / f"{args.name}.png"
    cv2.imwrite(str(png), patch)
    meta = {"threshold": args.threshold, "color": args.color}
    if args.search_around is not None:
        pad = args.search_around
        meta["region"] = [max(x1 - pad, 0), max(y1 - pad, 0), x2 + pad, y2 + pad]
    (ui.UI_TEMPLATE_DIR / f"{args.name}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {png} and {png.with_suffix('.json').name}")


def cmd_find(args):
    screen = _load_screen(args.screenshot)
    template = ui.load_template(args.name)
    best = ui.score(screen, template)
    if best is None:
        print("template is larger than the search area")
    else:
        verdict = "MATCH" if best.score >= template.threshold else "below threshold"
        print(
            f"{verdict}: score {best.score:.3f} (threshold {template.threshold}) "
            f"at {best.center}, click {best.click_point}"
        )


def cmd_blobs(args):
    from ..vision import find_color_blobs

    screen = _load_screen(args.screenshot)
    lower, upper = args.hsv[:3], args.hsv[3:]
    for x, y, w, h, area in find_color_blobs(screen, lower, upper, args.min_area)[:20]:
        print(f"blob at ({x}, {y}) size {w}x{h} area {area}")


def build_parser():
    parser = argparse.ArgumentParser(prog="stack")
    sub = parser.add_subparsers(dest="cmd", required=True)

    snap = sub.add_parser("snap", help="save a normalised 1920x1080 screenshot")
    snap.add_argument("output")
    snap.set_defaults(func=cmd_snap)

    crop = sub.add_parser("crop", help="save part of a screenshot as a UI template")
    crop.add_argument("screenshot")
    crop.add_argument("name")
    crop.add_argument("box", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"))
    crop.add_argument("--threshold", type=float, default=ui.DEFAULT_THRESHOLD)
    crop.add_argument("--color", action="store_true", help="match in colour, not grayscale")
    crop.add_argument(
        "--search-around",
        type=int,
        metavar="PX",
        help="only search within PX pixels of where it was cropped",
    )
    crop.set_defaults(func=cmd_crop)

    find = sub.add_parser("find", help="score a template against a screenshot")
    find.add_argument("screenshot")
    find.add_argument("name")
    find.set_defaults(func=cmd_find)

    blobs = sub.add_parser("blobs", help="list blobs inside an HSV range")
    blobs.add_argument("screenshot")
    blobs.add_argument("hsv", nargs=6, type=int, metavar=("H1", "S1", "V1", "H2", "S2", "V2"))
    blobs.add_argument("--min-area", type=int, default=200)
    blobs.set_defaults(func=cmd_blobs)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
