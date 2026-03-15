from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


# Configuration

NUMBER_OF_DIMENSIONS = 3
DISPLAYED_DIMENSIONS = [0, 1, 2]  # 0 indexed

# Get Points


def generate_n_dimensional_ball_points(
    num_dims: int,
    num_points: int,
) -> np.ndarray:
    points = np.random.normal(size=(num_points, num_dims))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    radii = np.random.uniform(0, 1, size=(num_points, 1)) ** (1 / num_dims)
    return points * radii


def generate_chaos_game_fractal(
    vertices: np.ndarray, num_points: int, step_size: float = 0.5
) -> np.ndarray:
    num_vertices, vertices_dimension = vertices.shape
    fractal_points = np.zeros((num_points, vertices_dimension))
    current_point = vertices[0]
    for i in range(num_points):
        random_vertex = vertices[np.random.randint(num_vertices)]
        current_point = current_point + step_size * (random_vertex - current_point)
        fractal_points[i] = current_point
    return fractal_points


def subspace_project(points: np.ndarray, orthonormal_basis: np.ndarray) -> np.ndarray:
    return points @ orthonormal_basis.T


def slice_project(
    points: np.ndarray,
    orthonormal_basis: np.ndarray,
    tolerance: float,
    offset: np.ndarray | None = None,
) -> np.ndarray:
    if offset is None:
        offset = np.zeros(points.shape[1])
    projection_matrix = orthonormal_basis.T @ orthonormal_basis
    # A.T @ A instead of standard A @ A.T since points are stored in rows
    translated = points - offset
    projected = translated @ projection_matrix
    residual = translated - projected
    distance = np.linalg.norm(residual, axis=1)
    mask = distance < tolerance
    return points[mask]


def stereographic_project(points: np.ndarray) -> np.ndarray:
    denominator = 1 - points[:, -1]
    if np.any(np.abs(denominator) < 1e-9):
        raise ValueError(
            "Stereographic Projection undefined for points where last dimension equals one"
        )
    return points[:, :-1] / denominator[:, np.newaxis]


def project_points_to_three_dimensions(
    points: np.ndarray, dimensions: list[int], projection_type: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.shape[1] < 3:
        padding = np.zeros((points.shape[0], 3 - points.shape[1]))
        points = np.hstack((points, padding))

    if projection_type == "subspace_project":
        orthonormal_basis = np.eye(points.shape[1])[dimensions]
        projected = subspace_project(points, orthonormal_basis)
    elif projection_type == "slice_project":
        sliced = slice_project(points, orthonormal_basis, tolerance=0.2)
        projected = subspace_project(sliced, orthonormal_basis)
    elif projection_type == "stereographic_project":
        projected = points
        while projected.shape[1] > 3:
            projected = stereographic_project(projected)
    else:
        raise ValueError(f"Unknown projection type: {projection_type}")

    x, y, z = projected.T
    return x, y, z


sierpinski_triangle_vertices = (
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

points = generate_chaos_game_fractal(sierpinski_triangle_vertices, 10000)
points = slice_project(points, np.eye(3, 4), 0.2, np.array([0, 0, 0, 1.25]))

# Scatter Plot Settings

BACKGROUND_COLOR = "#111111"

fig = plt.figure()
ax: Axes3D = fig.add_subplot(projection="3d")

ax.scatter(
    *project_points_to_three_dimensions(
        points, dimensions=DISPLAYED_DIMENSIONS, projection_type="stereographic_project"
    ),
    s=1,
)


ax.set_aspect("equal")
ax.set_axis_off()
fig.patch.set_facecolor(BACKGROUND_COLOR)
ax.set_facecolor(BACKGROUND_COLOR)

plt.show()
