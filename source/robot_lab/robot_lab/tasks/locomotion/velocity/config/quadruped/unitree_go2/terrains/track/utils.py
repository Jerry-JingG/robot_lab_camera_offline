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