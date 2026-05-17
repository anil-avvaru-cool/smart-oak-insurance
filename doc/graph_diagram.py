"""Generate graph_model.png — Neo4j schema diagram for smart-oak-insurance."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── colour palette ────────────────────────────────────────────────────────────
BG          = "#0d1117"
NODE_CLAIM  = "#1f6feb"   # blue
NODE_CLT    = "#238636"   # green
NODE_ATT    = "#da3633"   # red
NODE_VEH    = "#9e6a03"   # amber
NODE_ENT    = "#6e40c9"   # purple
EDGE_COLOR  = "#8b949e"
LABEL_COLOR = "#e6edf3"
PROP_COLOR  = "#8b949e"
FRAUD_RING  = "#ff7b72"

fig, ax = plt.subplots(figsize=(20, 13), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 20)
ax.set_ylim(0, 13)
ax.axis("off")

# ── node positions ────────────────────────────────────────────────────────────
#   Claim (centre), Claimant (left-centre), Attorney (top-left),
#   Vehicle (bottom-centre), Entity (right — two instances: phone & address)
POS = {
    "Claim":     (10.0, 6.5),
    "Claimant":  (5.5,  6.5),
    "Attorney":  (2.5,  9.5),
    "Vehicle":   (10.0, 2.8),
    "Entity_ph": (15.5, 9.0),
    "Entity_ad": (15.5, 4.5),
}

NODE_W, NODE_H = 2.8, 1.6   # box width / height

NODE_META = {
    "Claim": {
        "color": NODE_CLAIM,
        "label": "Claim",
        "props": [
            "id (unique)",
            "is_fraud: bool",
            "state: str",
            "shared_attribute_count",
            "graph_hop_distance",
            "attorney_centrality",
        ],
    },
    "Claimant": {
        "color": NODE_CLT,
        "label": "Claimant",
        "props": [
            "id (unique)",
            "name: str",
            "phone: str",
            "address: str",
            "is_fraud: bool",
        ],
    },
    "Attorney": {
        "color": NODE_ATT,
        "label": "Attorney",
        "props": [
            "id (unique)",
            "name: str",
            "centrality: float",
        ],
    },
    "Vehicle": {
        "color": NODE_VEH,
        "label": "Vehicle",
        "props": [
            "vin (unique)",
            "make: str",
            "model: str",
            "year: int",
        ],
    },
    "Entity_ph": {
        "color": NODE_ENT,
        "label": "Entity",
        "props": [
            "id (unique)",
            'type = "phone"',
            "value: str",
        ],
    },
    "Entity_ad": {
        "color": NODE_ENT,
        "label": "Entity",
        "props": [
            "id (unique)",
            'type = "address"',
            "value: str",
        ],
    },
}


def draw_node(ax, key, cx, cy, meta):
    """Draw a rounded-rect node with title bar + property list."""
    w, h_title = NODE_W, 0.55
    prop_h = 0.30
    total_h = h_title + len(meta["props"]) * prop_h + 0.15
    x0 = cx - w / 2
    y0 = cy - total_h / 2

    # shadow
    shadow = FancyBboxPatch(
        (x0 + 0.07, y0 - 0.07), w, total_h,
        boxstyle="round,pad=0.08", linewidth=0,
        facecolor="black", alpha=0.45, zorder=2,
    )
    ax.add_patch(shadow)

    # title bar
    title_box = FancyBboxPatch(
        (x0, y0 + total_h - h_title), w, h_title,
        boxstyle="round,pad=0.0", linewidth=0,
        facecolor=meta["color"], zorder=3,
    )
    ax.add_patch(title_box)

    # body
    body_box = FancyBboxPatch(
        (x0, y0), w, total_h - h_title,
        boxstyle="round,pad=0.0", linewidth=0,
        facecolor="#161b22", zorder=3,
    )
    ax.add_patch(body_box)

    # border
    border = FancyBboxPatch(
        (x0, y0), w, total_h,
        boxstyle="round,pad=0.0", linewidth=1.5,
        edgecolor=meta["color"], facecolor="none", zorder=4,
    )
    ax.add_patch(border)

    # title text
    ax.text(
        cx, y0 + total_h - h_title / 2,
        f":{meta['label']}",
        ha="center", va="center",
        fontsize=11, fontweight="bold",
        color="white", zorder=5,
    )

    # properties
    for i, prop in enumerate(meta["props"]):
        py = y0 + total_h - h_title - (i + 0.75) * prop_h
        ax.text(
            x0 + 0.15, py, prop,
            ha="left", va="center",
            fontsize=7.5, color=PROP_COLOR,
            fontfamily="monospace", zorder=5,
        )

    # return centre-y of box for edge anchoring
    return cy, y0, y0 + total_h


def arrow(ax, src, dst, label, offset=(0, 0), color=EDGE_COLOR,
          lw=1.6, style="arc3,rad=0.0", dashes=None):
    """Draw a labelled directed arrow between two (x,y) points."""
    sx, sy = src[0] + offset[0], src[1] + offset[1]
    dx, dy = dst[0] + offset[0], dst[1] + offset[1]
    arrowprops = dict(
        arrowstyle="-|>",
        lw=lw,
        color=color,
        connectionstyle=style,
    )
    if dashes:
        arrowprops["linestyle"] = (0, dashes)
    ax.annotate(
        "", xy=(dx, dy), xytext=(sx, sy),
        arrowprops=arrowprops, zorder=6,
    )
    mx, my = (sx + dx) / 2, (sy + dy) / 2
    ax.text(
        mx, my, label,
        ha="center", va="center",
        fontsize=8, color=color,
        fontweight="bold",
        bbox=dict(
            facecolor=BG, edgecolor="none",
            boxstyle="round,pad=0.15", alpha=0.85,
        ),
        zorder=7,
    )


# ── draw nodes ────────────────────────────────────────────────────────────────
box_bounds = {}
for key, (cx, cy) in POS.items():
    _, y0, y1 = draw_node(ax, key, cx, cy, NODE_META[key])
    box_bounds[key] = (cx, cy, y0, y1)


def mid_right(key):
    cx, cy, y0, y1 = box_bounds[key]
    return (cx + NODE_W / 2, (y0 + y1) / 2)


def mid_left(key):
    cx, cy, y0, y1 = box_bounds[key]
    return (cx - NODE_W / 2, (y0 + y1) / 2)


def mid_top(key):
    cx, cy, y0, y1 = box_bounds[key]
    return (cx, y1)


def mid_bottom(key):
    cx, cy, y0, y1 = box_bounds[key]
    return (cx, y0)


# ── edges ─────────────────────────────────────────────────────────────────────

# Claimant -> Claim [:FILED]
arrow(ax, mid_right("Claimant"), mid_left("Claim"), ":FILED",
      color="#58a6ff", style="arc3,rad=0.15")

# Claimant -> Attorney [:REPRESENTED_BY]
arrow(ax, mid_top("Claimant"), mid_bottom("Attorney"), ":REPRESENTED_BY",
      color=NODE_ATT, style="arc3,rad=-0.2")

# Claim -> Vehicle [:INVOLVES]
arrow(ax, mid_bottom("Claim"), mid_top("Vehicle"), ":INVOLVES",
      color=NODE_VEH, style="arc3,rad=0.0")

# Claimant -> Entity_ph [:SHARES {type:"phone"}]
arrow(ax, mid_right("Claimant"),
      (POS["Entity_ph"][0] - NODE_W / 2, POS["Entity_ph"][1] + 0.4),
      ':SHARES\n{type:"phone"}',
      color=NODE_ENT, style="arc3,rad=-0.25")

# Claim -> Entity_ph [:SHARES {type:"phone"}]
arrow(ax, mid_right("Claim"),
      (POS["Entity_ph"][0] - NODE_W / 2, POS["Entity_ph"][1] - 0.1),
      ':SHARES\n{type:"phone"}',
      color=NODE_ENT, style="arc3,rad=0.2")

# Claimant -> Entity_ad [:SHARES {type:"address"}]
arrow(ax, mid_right("Claimant"),
      (POS["Entity_ad"][0] - NODE_W / 2, POS["Entity_ad"][1] + 0.5),
      ':SHARES\n{type:"address"}',
      color="#a371f7", style="arc3,rad=0.3")

# Claim -> Entity_ad [:SHARES {type:"address"}]
arrow(ax, mid_right("Claim"),
      (POS["Entity_ad"][0] - NODE_W / 2, POS["Entity_ad"][1] - 0.2),
      ':SHARES\n{type:"address"}',
      color="#a371f7", style="arc3,rad=-0.15")

# ── fraud ring callout ────────────────────────────────────────────────────────
ax.text(
    15.5, 11.2,
    "Fraud Ring Detection",
    ha="center", va="center",
    fontsize=12, fontweight="bold",
    color=FRAUD_RING,
)
ax.text(
    15.5, 10.75,
    "Multiple Claimants / Claims sharing\nthe same Entity node → fraud signal",
    ha="center", va="center",
    fontsize=8.5, color=PROP_COLOR,
    linespacing=1.5,
)
ax.plot([15.5], [10.35], marker="v", ms=10, color=FRAUD_RING, zorder=8)

# ── BFS feature callout ───────────────────────────────────────────────────────
ax.text(
    10.0, 12.3,
    "Graph Features (written back to claims.parquet)",
    ha="center", va="center",
    fontsize=10, fontweight="bold", color="#58a6ff",
)
for i, feat in enumerate([
    "graph_hop_distance  — BFS hops from nearest fraud Claim (sentinel 999 = unreachable)",
    "shared_attribute_count  — # distinct shared Entity nodes across Claims",
    "attorney_centrality_score  — normalised degree of Claimant's Attorney",
]):
    ax.text(
        10.0, 11.85 - i * 0.38,
        f"• {feat}",
        ha="center", va="center",
        fontsize=7.5, color=PROP_COLOR,
        fontfamily="monospace",
    )

# ── title ─────────────────────────────────────────────────────────────────────
ax.text(
    10.0, 0.45,
    "Smart Oak Insurance — Neo4j Graph Model",
    ha="center", va="center",
    fontsize=14, fontweight="bold", color=LABEL_COLOR,
)
ax.text(
    10.0, 0.12,
    "Nodes: Claim · Claimant · Attorney · Vehicle · Entity    "
    "   Fraud ring detection via shared Entity nodes",
    ha="center", va="center",
    fontsize=8, color=PROP_COLOR,
)

# ── legend ────────────────────────────────────────────────────────────────────
legend_items = [
    (NODE_CLAIM, ":Claim"),
    (NODE_CLT,   ":Claimant"),
    (NODE_ATT,   ":Attorney"),
    (NODE_VEH,   ":Vehicle"),
    (NODE_ENT,   ":Entity"),
]
for i, (color, lbl) in enumerate(legend_items):
    rx = 0.5 + i * 3.8
    patch = mpatches.FancyBboxPatch(
        (rx, 0.6), 0.45, 0.28,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor="none", zorder=5,
    )
    ax.add_patch(patch)
    ax.text(rx + 0.6, 0.74, lbl, va="center", fontsize=8,
            color=LABEL_COLOR, zorder=6)

# ── save ──────────────────────────────────────────────────────────────────────
out = "doc/graph_model.png"
plt.tight_layout(pad=0)
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
