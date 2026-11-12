"""Side-by-side figure of the figure frames (protocol.FIGURE_FRAMES): one row
per frame, the observed image first, then one column per method, from the
figures/ folders written by score.py and from_monogs_record.py --figures.

    python baselines/common/compare_figures.py --out figures_office0_lit.png \
        --column "Đồ án=<dir>/figures" --column "SplaTAM=<dir>/figures" ...

Images are pasted unscaled (all are at the dataset resolution); labels on
top. Use --frames to pick a subset of the figure frames.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import protocol  # noqa: E402


def label_font(label_px):
    """DejaVu Sans from matplotlib (Vietnamese glyphs); PIL's default bitmap
    font has none."""
    from matplotlib import font_manager
    from PIL import ImageFont

    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), max(8, int(label_px * 0.7)))


def main(argv=None):
    from PIL import Image, ImageDraw

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--column", action="append", required=True, help="label=figures folder")
    ap.add_argument("--frames", type=int, nargs="*", default=list(protocol.FIGURE_FRAMES))
    ap.add_argument("--label_px", type=int, default=28, help="height of the label band")
    args = ap.parse_args(argv)
    cols = []
    for c in args.column:
        label, _, path = c.rpartition("=")  # the label may contain "="
        if not path:
            raise SystemExit(f"--column needs label=folder, got {c!r}")
        cols.append((label, path))
    first = cols[0][1]
    obs = [Image.open(os.path.join(first, f"{i:06d}_obs.jpg")).convert("RGB") for i in args.frames]
    w, h = obs[0].size
    grid = Image.new("RGB", (w * (len(cols) + 1), args.label_px + h * len(args.frames)), "white")
    draw = ImageDraw.Draw(grid)
    font = label_font(args.label_px)
    for j, label in enumerate(["Quan sát"] + [c[0] for c in cols]):
        draw.text((j * w + 6, 4), label, fill="black", font=font)
    for r, i in enumerate(args.frames):
        y = args.label_px + r * h
        grid.paste(obs[r], (0, y))
        draw.text((6, y + 4), f"#{i}", fill="white", font=font)
        for j, (_, path) in enumerate(cols, start=1):
            p = os.path.join(path, f"{i:06d}_render.png")
            if not os.path.isfile(p):
                raise SystemExit(f"missing {p}")
            grid.paste(Image.open(p).convert("RGB"), (j * w, y))
    grid.save(args.out)
    print(f"{len(args.frames)} frames x {len(cols)} methods -> {args.out}")


if __name__ == "__main__":
    main()
