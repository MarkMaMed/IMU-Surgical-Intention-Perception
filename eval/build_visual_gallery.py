from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight HTML gallery for generated plots.")
    parser.add_argument("--plots-dir", default="plots")
    parser.add_argument("--output", default="plots/gallery.html")
    args = parser.parse_args()

    plots_dir = Path(args.plots_dir)
    if not plots_dir.exists():
        raise FileNotFoundError(f"Plots directory not found: {plots_dir}")

    images = sorted(
        [p for p in plots_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and p.is_file()]
    )
    cards = []
    for p in images:
        rel = p.relative_to(plots_dir)
        cards.append(
            f"""
            <div class="card">
              <h3>{rel}</h3>
              <img src="{rel.as_posix()}" alt="{rel}">
            </div>
            """
        )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>IMU 可视化图库</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f6f8fb; }}
    h1 {{ margin-bottom: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill,minmax(360px,1fr)); gap: 16px; }}
    .card {{ background: #fff; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.08); padding: 12px; }}
    .card h3 {{ margin: 0 0 8px 0; font-size: 14px; color: #334; }}
    .card img {{ width: 100%; border-radius: 6px; border: 1px solid #e8ecf3; }}
  </style>
</head>
<body>
  <h1>IMU 意态感知可视化图库</h1>
  <p>图像数量: {len(images)}</p>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Saved gallery: {out}")


if __name__ == "__main__":
    main()

