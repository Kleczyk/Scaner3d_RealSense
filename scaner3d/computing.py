import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import os
from datetime import datetime


def frame_to_point_cloud(depth_frame, frames):
    """
    Konwertuje ramkę głębi do chmury punktów.
    :param depth_frame: ramka głębi
    :return: chmura punktów
    """
    pc = rs.pointcloud()
    print(type(pc))
    pc.map_to(frames.get_color_frame())
    points = pc.calculate(depth_frame)
    vtx = np.asanyarray(points.get_vertices())
    vtx_np = np.array([(v[0], v[1], v[2]) for v in vtx], dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vtx_np)
    return pcd

def write_point_cloud_to_file(pcd_list):
    """
    Zapisuje chmurę punktów do pliku .ply.
    :param pcd_list: lista chmur punktów
    :return: None

    """

    # Pobieranie aktualnej daty i godziny
    current_time = datetime.now()

    # Formatowanie daty i godziny do postaci string (np. "2024-01-03_15-30-00" dla 3 stycznia 2024, 15:30:00)
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Możesz dodać bazowy katalog przed nazwą folderu, jeśli chcesz
    base_directory = "ply_files"
    folder_path = os.path.join(base_directory, folder_name)

    # Tworzenie folderu
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' został utworzony.")
    else:
        print(f"Folder '{folder_path}' już istnieje.")

    # Zapis chmury punktów do pliku
    for i, pcd in enumerate(pcd_list):
        filename = os.path.join(folder_path, f"point_cloud_{i}.ply")
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Chmura punktów została zapisana do pliku '{filename}'.")
