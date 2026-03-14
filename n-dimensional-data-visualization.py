from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


# Configuration

NUMBER_OF_DIMENSIONS = 3
DISPLAYED_DIMENSIONS = [0, 1, 2]  # 0 indexed

# Get Points


def generate_n_dimensional_ball_points(n: int, num_points: int) -> np.ndarray:
    points = np.random.normal(size=(num_points, n))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    radii = np.random.uniform(0, 1, size=(num_points, 1)) ** (1 / n)
    return points * radii


def subspace_project(points: np.ndarray, orthonormal_basis: np.ndarray) -> np.ndarray:
    return points @ orthonormal_basis.T


def slice_project(
    points: np.ndarray, orthonormal_basis: np.ndarray, tolerance: float
) -> np.ndarray:
    projection_matrix = orthonormal_basis.T @ orthonormal_basis
    projected = points @ projection_matrix
    residual = points - projected
    distance = np.linalg.norm(residual, axis=1)
    mask = distance < tolerance
    return points[mask]


def project_points_to_three_dimensions(
    points: np.ndarray, dimensions: list[int], projection_type: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    orthonormal_basis = np.eye(points.shape[1])[dimensions]
    if projection_type == "subspace_project":
        projected = subspace_project(points, orthonormal_basis)
    elif projection_type == "slice_project":
        sliced = slice_project(points, orthonormal_basis, tolerance=0.2)
        projected = subspace_project(sliced, orthonormal_basis)
    else:
        raise ValueError(f"Unknown projection type: {projection_type}")

    x, y, z = projected.T
    return x, y, z


# Scatter Plot Settings

BACKGROUND_COLOR = "#111111"

fig = plt.figure()
ax: Axes3D = fig.add_subplot(projection="3d")

points = generate_n_dimensional_ball_points(NUMBER_OF_DIMENSIONS, 1000)
ax.scatter(
    *project_points_to_three_dimensions(
        points, dimensions=DISPLAYED_DIMENSIONS, projection_type="slice_project"
    )
)

ax.set_aspect("equal")
ax.set_axis_off()

fig.patch.set_facecolor(BACKGROUND_COLOR)
ax.set_facecolor(BACKGROUND_COLOR)

plt.show()
