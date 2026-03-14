import numpy as np
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────
N_DIMS = 4
N_POINTS = 8000

# ── Simplex vertices ──────────────────────────────────────────


def regular_simplex_vertices(dims: int, scale: float = 1.0) -> np.ndarray:
    """Vertices of a regular simplex embedded in `dims` dimensions."""
    verts = np.zeros((dims + 1, dims))
    for i in range(1, dims + 1):
        verts[i, :i] = 1
        verts[i, i - 1] = -i / np.sqrt(i * (i + 1))
    verts -= verts.mean(axis=0)
    max_edge = np.max(np.linalg.norm(verts[:, None] - verts[None, :], axis=-1))
    return verts * scale / max_edge


# ── Chaos game ────────────────────────────────────────────────


def chaos_game(vertices: np.ndarray, num_points: int) -> np.ndarray:
    """
    Play the chaos game on a set of vertices in n-dimensional space.
    At each step, move halfway from the current point to a randomly chosen vertex.
    Returns the resulting point cloud.
    """
    n_verts, dims = vertices.shape
    points = np.zeros((num_points, dims))
    p = vertices[0].copy()
    for i in range(num_points):
        v = vertices[np.random.randint(n_verts)]
        p = (p + v) / 2
        points[i] = p
    return points


# ── Stereographic projection ──────────────────────────────────


def stereographic(points: np.ndarray) -> np.ndarray:
    """
    Project 4D points into 3D by casting rays from the pole at w=2.
    Each point (x, y, z, w) maps to (x, y, z) / (2 - w).
    Points near the pole (w ≈ 2) are discarded.
    """
    x, y, z, w = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    denom = 1 - w
    safe = np.abs(denom) > 1e-9
    d = denom[safe]
    return np.column_stack((x[safe] / d, y[safe] / d, z[safe] / d))


# ── Generate data ─────────────────────────────────────────────

verts = regular_simplex_vertices(N_DIMS)
points = chaos_game(verts, N_POINTS)
proj = stereographic(points)

# ── Plot ──────────────────────────────────────────────────────

BG = "#0d0d0d"
FG = "#cccccc"

fig = plt.figure(figsize=(8, 8))
fig.patch.set_facecolor(BG)

ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor(BG)
ax.xaxis.pane.fill = False  # type: ignore[attr-defined]
ax.yaxis.pane.fill = False  # type: ignore[attr-defined]
ax.zaxis.pane.fill = False  # type: ignore[attr-defined]
ax.xaxis.pane.set_edgecolor("#222222")  # type: ignore[attr-defined]
ax.yaxis.pane.set_edgecolor("#222222")  # type: ignore[attr-defined]
ax.zaxis.pane.set_edgecolor("#222222")  # type: ignore[attr-defined]
ax.tick_params(colors=FG, labelsize=7)

ax.scatter(
    proj[:, 0], proj[:, 1], proj[:, 2], c="mediumpurple", s=4, alpha=0.5, linewidths=0
)

plt.tight_layout()
plt.show()
