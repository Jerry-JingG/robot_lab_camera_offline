import numpy as np
import scipy.spatial.transform as tf
import trimesh


def make_overhangs_box(
    length: float,
    width: float,
    height: float,
    center: tuple[float, float, float],
    max_yx_angle: float = 0,
    max_z_angle: float = 0,
    degrees: bool = True,
) -> trimesh.Trimesh:
    """Generate a box mesh with a random orientation.

    Args:
        length: The length (along x) of the box (in m).
        width: The width (along y) of the box (in m).
        height: The height of the cylinder (in m).
        center: The center of the cylinder (in m).
        max_yx_angle: The maximum angle along the y and x axis. Defaults to 0.
        max_z_angle: The maximum angle along the z axis. Defaults to 0.
        degrees: Whether the angle is in degrees. Defaults to True.

    Returns:
        A trimesh.Trimesh object for the cylinder.
    """
    # create a pose for the cylinder
    transform = np.eye(4)
    transform[0:3, -1] = np.asarray(center)
    # -- create a random rotation
    euler_zyx = tf.Rotation.random().as_euler("zyx")  # returns rotation of shape (3,)
    # -- cap the rotation along the y and x axis
    if degrees:
        max_yx_angle = max_yx_angle / 180.0
        max_z_angle = max_z_angle / 180.0
    euler_zyx[1:] *= max_yx_angle
    euler_zyx[0] *= max_z_angle
    # -- apply the rotation
    transform[0:3, 0:3] = tf.Rotation.from_euler("zyx", euler_zyx).as_matrix()
    # create the box
    dims = (length, width, height)
    return trimesh.creation.box(dims, transform=transform)


def make_narrow_crack(
    length: float,
    width: float,
    height: float,
    center: tuple[float, float, float],
    gap_x_offset: float = 0.0,
    gap_width: float = 1.0,
) -> list[trimesh.Trimesh]:
    """Generate a narrow gap obstacle that robot can pass through along y-axis.

    Creates two box meshes on either side of a narrow gap, allowing the robot
    to traverse through the gap in the positive y direction.

    Args:
        length: The total length (along x) of the obstacle (in m).
        width: The total width (along y) of the obstacle (in m).
        height: The height of the obstacle (in m).
        center: The center of the entire obstacle (in m).
        gap_x_offset: The x-axis offset of the gap from center. Positive moves gap to +x direction.
        gap_width: The width of the gap (in m).

    Returns:
        A list of trimesh.Trimesh objects representing the two sides of the gap.
    """
    meshes = []

    # Calculate the center position
    center_x, center_y, center_z = center

    # Calculate gap position
    gap_center_x = center_x + gap_x_offset

    # Calculate the width of each side box
    # Left side: from start to gap start
    left_width = gap_center_x - gap_width / 2 - (center_x - length / 2)
    # Right side: from gap end to obstacle end
    right_width = (center_x + length / 2) - (gap_center_x + gap_width / 2)

    # Only create boxes if they have positive width
    if left_width > 0:
        # Left side box
        left_center_x = center_x - length / 2 + left_width / 2
        left_transform = np.eye(4)
        left_transform[0:3, -1] = np.array([left_center_x, center_y, center_z])
        left_dims = (left_width, width, height)
        left_box = trimesh.creation.box(left_dims, transform=left_transform)
        meshes.append(left_box)

    if right_width > 0:
        # Right side box
        right_center_x = center_x + length / 2 - right_width / 2
        right_transform = np.eye(4)
        right_transform[0:3, -1] = np.array([right_center_x, center_y, center_z])
        right_dims = (right_width, width, height)
        right_box = trimesh.creation.box(right_dims, transform=right_transform)
        meshes.append(right_box)

    return meshes


def make_highland_platform(
    length: float,
    width: float,
    height: float,
    center: tuple[float, float, float],
) -> list[trimesh.Trimesh]:
    """Generate a highland platform obstacle that robot can climb up and down.

    Creates a raised platform that robot needs to jump or climb over.

    Args:
        length: The total length (along x) of the platform (in m).
        width: The total width (along y) of the platform (in m).
        height: The height of the platform (in m).
        center: The center of the entire platform (in m).

    Returns:
        A list of trimesh.Trimesh objects representing the platform.
    """
    meshes = []

    center_x, center_y, center_z = center

    # Main platform
    platform_center_z = center_z + height / 2.0  # Place on top of ground

    platform_transform = np.eye(4)
    platform_transform[0:3, -1] = np.array([center_x, center_y, platform_center_z])
    platform_dims = (length, width, height)
    platform_box = trimesh.creation.box(platform_dims, transform=platform_transform)
    meshes.append(platform_box)

    return meshes