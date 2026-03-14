import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # for 3D scatter
import sklearn as sklearn
from sklearn.decomposition import PCA

n = 4  # number of total dimensions
fixed_dims = [0, 1, 2]  # will show these dimensions
slider_dims = [d for d in range(n) if d not in fixed_dims]


def pad_points_to_n(points, target_n):
    current_n = points.shape[1]
    if current_n < target_n:
        pad_width = target_n - current_n
        pad = np.zeros((points.shape[0], pad_width))
        points = np.hstack((points, pad))
    return points


def generate_hypersphere_points(n, num_points, shell=True):
    """
    Generate points on or inside an n-dimensional hypersphere.
    If shell=True, points lie on the surface; otherwise, inside the volume.
    """
    points = np.random.normal(size=(num_points, n))
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    if not shell:
        radii = np.random.rand(num_points) ** (1.0 / n)
        points *= radii[:, np.newaxis]
    return points


# Generate hypersphere points
hypersphere_points = generate_hypersphere_points(n, 1000)


def project_points_pca(points, end_dim):
    pca = PCA(n_components=end_dim)
    return pca.fit_transform(points)


hypersphere_points = project_points_pca(hypersphere_points, 3)

sierpinski_vertices = (
    np.array(
        [
            [1, 1, 1, -1 / np.sqrt(5)],
            [1, -1, -1, -1 / np.sqrt(5)],
            [-1, 1, -1, -1 / np.sqrt(5)],
            [-1, -1, 1, -1 / np.sqrt(5)],
            [0, 0, 0, 4 / np.sqrt(5)],
        ]
    )
    / 2
)


def generate_random_walk_fractal(vertices, num_points, n, step_size=0.5):
    """
    Generate fractal points using random walk between given vertices.

    Args:
        vertices: np.array of shape (num_vertices, vertex_dim) - Initial vertices
        num_points: int - Number of points to generate
        n: int - Target dimension to pad to
        step_size: float - Fraction of distance to move (default 0.5 for classical fractals)

    Returns:
        np.array of shape (num_points, n) - Generated points padded to n dimensions
    """
    # Validate input
    if not isinstance(vertices, np.ndarray):
        vertices = np.array(vertices)

    num_vertices = vertices.shape[0]
    vertex_dim = vertices.shape[1]

    # Initialize points array
    points = np.zeros((num_points, vertex_dim))

    # Start from random point
    current_point = np.random.rand(vertex_dim)
    points[0] = current_point

    # Random walk
    for i in range(1, num_points):
        # Choose random vertex
        target_vertex = vertices[np.random.randint(0, num_vertices)]
        # Move towards vertex
        current_point = current_point + step_size * (target_vertex - current_point)
        points[i] = current_point

    # Pad to n dimensions
    points = pad_points_to_n(points, n)

    return points


# ---- Generate 3D curve----
xs = np.linspace(0, 1, 100)
ys = np.sin(xs * 6 * np.pi)
zs = np.cos(xs * 6 * np.pi)

curve_points = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
curve_points = pad_points_to_n(curve_points, n)

# ---- Generate 4D surface: z = x^2 + y^2 + w^2 ----
grid_size = 50
x_vals = np.linspace(-5, 5, grid_size)
y_vals = np.linspace(-5, 5, grid_size)
X, Y = np.meshgrid(x_vals, y_vals)
X_flat = X.ravel()
Y_flat = Y.ravel()
# For w dimension, initially set to zero; will be adjusted in update()
W_flat = np.zeros_like(X_flat)
Z_flat = X_flat**2 + Y_flat**2 + W_flat**2
quad_surface_points = np.stack([X_flat, Y_flat, Z_flat, W_flat], axis=1)

"""
# ---- Generate 3D surface----
x_vals = np.linspace(-5, 5, 100)
y_vals = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.sin(X) * np.cos(Y)

# ---- Generate 3D solid----
# r = 1.0
# n_vox = 20
# grid = np.linspace(-r, r, n_vox)
# Xv, Yv, Zv = np.meshgrid(grid, grid, grid, indexing='ij')

# 2) mask of voxels inside the unit sphere
# voxel_mask = (Xv[:-1, :-1, :-1]**2 + Yv[:-1, :-1, :-1]**2 + Zv[:-1, :-1, :-1]**2) <= r**2
# print(Xv.shape, Yv.shape, Zv.shape, voxel_mask.shape)
"""

# Set up figure
fig = plt.figure()
fig.patch.set_facecolor("#111111")  # Set figure background
plt.subplots_adjust(left=0.1, bottom=0.25)

# Choose correct axis type
if len(fixed_dims) == 1:
    ax = fig.add_subplot(111)
    plot_type = "1d"
elif len(fixed_dims) == 2:
    ax = fig.add_subplot(111)
    plot_type = "2d"
elif len(fixed_dims) == 3:
    ax = fig.add_subplot(111, projection="3d")
    plot_type = "3d"
else:
    raise ValueError("Can only visualize 1D, 2D, or 3D plots with matplotlib.")

# Define color variables
background_color = "#111111"
# grid_color = "#670D2F"
# tick_color = "#A53860"
# spine_color = "#EF88AD"

# Set plot background and grid/tick/spine colors for dark mode
# ax.grid(color=grid_color)
# ax.tick_params(colors=tick_color)
# for spine in ax.spines.values():
#    spine.set_color(spine_color)

ax.set_axis_off()
ax.set_facecolor(background_color)


# if plot_type == "3d":
#     # 3D axes don't have spines, but set tick label color
#     ax.xaxis._axinfo["grid"]["color"] = grid_color
#     ax.yaxis._axinfo["grid"]["color"] = grid_color
#     ax.zaxis._axinfo["grid"]["color"] = grid_color
#     ax.xaxis.label.set_color(tick_color)
#     ax.yaxis.label.set_color(tick_color)
#     ax.zaxis.label.set_color(tick_color)
# else:
#     ax.xaxis.label.set_color(tick_color)
#     ax.yaxis.label.set_color(tick_color)

sierpinski_points = generate_random_walk_fractal(sierpinski_vertices, 10000, n)
sierpinski_vertices[:, 3] += 1

# Define shapes with points and colors
shapes = [
    # {"name": "curve", "points": curve_points, "color": "green", "artist": None},
    # {"name": "hypersphere", "points": hypersphere_points, "color": "red", "artist": None},
    {
        "name": "sierpinski",
        "points": generate_random_walk_fractal(sierpinski_vertices, 5000, n),
        "color": "cyan",
        "artist": None,
    },
]

# Initialize empty artists dynamically based on plot_type
if plot_type == "1d":
    for shape in shapes:
        if shape["name"] == "curve":
            (shape["artist"],) = ax.plot([], [], "-", color=shape["color"])
        else:
            (shape["artist"],) = ax.plot(
                [], [], "o", color=shape["color"], markersize=4, alpha=0.5
            )
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(f"Dim {fixed_dims[0]}")

elif plot_type == "2d":
    for shape in shapes:
        if shape["name"] == "curve":
            (shape["artist"],) = ax.plot([], [], "-", color=shape["color"])
        else:
            shape["artist"] = ax.scatter(
                [], [], c=shape["color"], s=5, marker="o", alpha=0.5
            )
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel(f"Dim {fixed_dims[0]}")
    ax.set_ylabel(f"Dim {fixed_dims[1]}")

elif plot_type == "3d":
    for shape in shapes:
        if shape["name"] == "curve":
            (shape["artist"],) = ax.plot(
                [], [], [], "-", color=shape["color"], linewidth=2
            )
        else:
            shape["artist"] = ax.scatter(
                [], [], [], c=shape["color"], s=2, marker="o", alpha=0.3
            )
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel(f"Dim {fixed_dims[0]}")
    ax.set_ylabel(f"Dim {fixed_dims[1]}")
    ax.set_zlabel(f"Dim {fixed_dims[2]}")

    ax.xaxis._axinfo["grid"]["alpha"] = 0.2
    ax.yaxis._axinfo["grid"]["alpha"] = 0.2
    ax.zaxis._axinfo["grid"]["alpha"] = 0.2

# Create sliders
sliders = []
for i, dim in enumerate(slider_dims):
    ax_slider = plt.axes([0.1, 0.15 - i * 0.04, 0.8, 0.03])
    slider = Slider(ax_slider, f"Dim {dim}", -5, 5, valinit=0)
    sliders.append(slider)


def project_points(points, fixed_dims, projection_type="orthographic"):
    """
    Projects the input points onto the dimensions specified in fixed_dims.

    Args:
        points: The input points in n-dimensional space.
        fixed_dims: The dimensions to project onto.
        projection_type: The type of projection to use ("orthographic" or "stereographic").

    Returns:
        For 1D: (x, y) where y is all zeros, x = points[:, fixed_dims[0]]
        For 2D: (x, y) as a tuple
        For 3D: (x, y, z) as a tuple
        For stereographic projection: (x, y, z) as a tuple from 4D to 3D.
    """
    if projection_type == "orthographic":
        if len(fixed_dims) == 1:
            x = points[:, fixed_dims[0]]
            y = np.zeros_like(x)
            return x, y
        elif len(fixed_dims) == 2:
            return (
                points[:, fixed_dims[0]],
                points[:, fixed_dims[1]],
            )
        elif len(fixed_dims) == 3:
            return (
                points[:, fixed_dims[0]],
                points[:, fixed_dims[1]],
                points[:, fixed_dims[2]],
            )
        else:
            raise ValueError(
                "Only 1D, 2D, or 3D orthographic projections are supported."
            )

    elif projection_type == "stereographic":
        if points.shape[1] < 4:
            raise ValueError("Stereographic projection requires at least 4D points.")

        # Stereographic projection from 4D to 3D
        x, y, z, w = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
        denominator = 1 - w  # Project from the north pole (w = 1)
        if np.any(denominator == 0):
            raise ValueError(
                "Stereographic projection undefined for points where w = 1."
            )

        x_proj = x / denominator
        y_proj = y / denominator
        z_proj = z / denominator
        return x_proj, y_proj, z_proj

    else:
        raise ValueError(
            "Unsupported projection type. Use 'orthographic' or 'stereographic'."
        )


def apply_nd_mask(points, slider_vals, slider_dims, atol=0.2):
    """
    Returns a boolean mask selecting points near the slider slice.
    """
    if len(slider_dims) == 0:
        return np.ones(points.shape[0], dtype=bool)
    return np.all(np.isclose(points[:, slider_dims], slider_vals, atol=atol), axis=1)


# Update function
def update(val):
    slider_vals = np.array([s.val for s in sliders])

    # Update the w coordinate for quad_surface_points (dimension 3)
    if 3 in slider_dims:
        w_index = slider_dims.index(3)
        current_w = slider_vals[w_index]
    else:
        current_w = 0.0
    quad_surface_points[:, 3] = current_w
    quad_surface_points[:, 2] = (
        quad_surface_points[:, 0] ** 2 + quad_surface_points[:, 1] ** 2 + current_w**2
    )

    for shape in shapes:
        points = shape["points"]
        mask = apply_nd_mask(points, slider_vals, slider_dims)
        visible_points = points[mask]

        artist = shape["artist"]
        if plot_type == "1d":
            if shape["name"] == "curve":
                if len(visible_points):
                    x, y = project_points(visible_points, fixed_dims)
                    artist.set_data(x, y)
                else:
                    artist.set_data([], [])
            else:
                if len(visible_points):
                    x, y = project_points(visible_points, fixed_dims)
                    artist.set_data(x, y)
                else:
                    artist.set_data([], [])

        elif plot_type == "2d":
            if shape["name"] == "curve":
                if len(visible_points):
                    x, y = project_points(visible_points, fixed_dims)
                    artist.set_data(x, y)
                else:
                    artist.set_data([], [])
            else:
                if len(visible_points):
                    x, y = project_points(visible_points, fixed_dims)
                    artist.set_offsets(np.column_stack((x, y)))
                else:
                    artist.set_offsets(np.empty((0, 2)))

        elif plot_type == "3d":
            if shape["name"] == "curve":
                if len(visible_points):
                    x, y, z = project_points(visible_points, fixed_dims)
                    artist.set_data_3d(x, y, z)
                else:
                    artist.set_data_3d([], [], [])
            else:
                if len(visible_points):
                    x, y, z = project_points(
                        visible_points, fixed_dims, projection_type="stereographic"
                    )
                    artist._offsets3d = (x, y, z)
                else:
                    artist._offsets3d = ([], [], [])

    fig.canvas.draw_idle()


for slider in sliders:
    slider.on_changed(update)

update(None)  # Initialize
plt.show()
