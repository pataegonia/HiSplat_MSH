"""HiSplat+MSH 헤드라인 RD 곡선 plot.

데이터: final_rd_curve.csv (10점) + vanilla 27.194 천장선.
"""

import csv
import matplotlib.pyplot as plt

CSV_PATH = "final_rd_curve.csv"
VANILLA_PSNR = 27.194  # HiSplat without codec (job 344145)


def load_rd_points(path: str) -> list[dict]:
    points = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            points.append({
                "lambda": float(row["lambda"]),
                "psnr":   float(row["psnr"]),
                "kb":     float(row["compressed_kb"]),
                "setup":  row["setup"],
            })
    # sort by KB ascending
    points.sort(key=lambda p: p["kb"])
    return points


points = load_rd_points(CSV_PATH)

fig, ax = plt.subplots(figsize=(9, 5.5))

kbs   = [p["kb"]     for p in points]
psnrs = [p["psnr"]   for p in points]
lams  = [p["lambda"] for p in points]

# RD curve
ax.plot(kbs, psnrs, marker="o", color="#1f77b4", linewidth=2,
        markersize=8, label="HiSplat + MSH (70k warmstart)")

# λ annotations
for x, y, l in zip(kbs, psnrs, lams):
    ax.annotate(f"λ={l:g}", xy=(x, y),
                xytext=(8, 6), textcoords="offset points",
                fontsize=9, color="#1f77b4")

# vanilla ceiling
ax.axhline(VANILLA_PSNR, color="#888", linestyle="--", linewidth=1.5,
           label=f"vanilla HiSplat (no codec) = {VANILLA_PSNR:.3f} dB")

ax.set_xlabel("Compressed size (KB)", fontsize=12)
ax.set_ylabel("PSNR (dB)", fontsize=12)
ax.set_title("HiSplat + MSH RD curve (re10k test, 7286 scenes)", fontsize=13)
ax.set_xlim(left=0)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=10)

plt.tight_layout()
plt.savefig("rd_curve.png", dpi=200)
plt.savefig("rd_curve.pdf")
print(f"Saved: rd_curve.png, rd_curve.pdf ({len(points)} points)")
