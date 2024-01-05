import pyrealsense2 as rs
import open3d as o3d
import numpy as np


def reconstruct_surface(pcd):
    """
    Rekonstruuje powierzchnię z chmury punktów.
    :param pcd: zarejestrowana chmura punktów
    :return: siatka (mesh) 3D
    """
    pcd.estimate_normals()
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9
    )[0]
    bbox = pcd.get_axis_aligned_bounding_box()
    return poisson_mesh.crop(bbox)
