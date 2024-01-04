import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import time
import os
from datetime import datetime
import plotly.graph_objs as go


def frame_to_point_cloud(depth_frame):
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


def register_point_clouds(pcd_list, max_distance=0.05):
    """
    Rejestruje chmury punktów z listy za pomocą ICP.
    :param pcd_list: lista chmur punktów
    :param max_distance: maksymalna odległość dla parowania punktów w ICP
    :return: zarejestrowana chmura punktów
    """
    registered_pcd = pcd_list[0]
    for pcd in pcd_list[1:]:
        # Przygotowanie do rejestracji
        source = pcd
        target = registered_pcd
        source.estimate_normals()
        target.estimate_normals()

        # Rejestracja ICP
        icp_coarse = o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_distance,
            np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=3),
        )

        # Transformacja chmury punktów
        source.transform(icp_coarse.transformation)

        # Połączenie zarejestrowanych chmur punktów
        registered_pcd = source + target

    return registered_pcd


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


def statistical_outlier_removal_open3d(pcd, k_neighbors=20, std_ratio=1.0):
    """
    Usuwa odstające punkty z chmury punktów za pomocą metody statystycznej.
    :param pcd: chmura punktów do przefiltrowania
    :param k_neighbors: liczba sąsiadów do analizy
    :param std_ratio: współczynnik odchylenia standardowego
    :return: przefiltrowana chmura punktów
    """
    filtered_pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=k_neighbors, std_ratio=std_ratio
    )
    return filtered_pcd


def filter_point_cloud_by_distance(pcd, min_distance, max_distance):
    # Konwertuje chmurę punktów na tablicę numpy
    points = np.asarray(pcd.points)

    # Oblicza odległość każdego punktu od początku układu współrzędnych (0,0,0)
    distances = np.linalg.norm(points, axis=1)

    # Filtruje punkty, które mieszczą się w określonym zakresie odległości
    filtered_points = points[(distances >= min_distance) & (distances <= max_distance)]

    # Tworzy nową chmurę punktów z odfiltrowanych punktów
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    return filtered_pcd


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


def crop_point_cloud(pcd, bounds):
    """
    Wycina fragment chmury punktów ograniczony do określonego prostopadłościanu.
    :param pcd: chmura punktów do przefiltrowania
    :param bounds: granice prostopadłościanu jako słownik z 'min' i 'max' dla każdej osi (x, y, z)
    :return: przefiltrowana chmura punktów
    """
    # Konwersja chmury punktów na tablicę numpy
    points = np.asarray(pcd.points)

    # Filtracja punktów w granicach
    in_bounds = (
        (points[:, 0] >= bounds["x_min"])
        & (points[:, 0] <= bounds["x_max"])
        & (points[:, 1] >= bounds["y_min"])
        & (points[:, 1] <= bounds["y_max"])
        & (points[:, 2] >= bounds["z_min"])
        & (points[:, 2] <= bounds["z_max"])
    )
    filtered_points = points[in_bounds]

    # Tworzenie nowej chmury punktów z odfiltrowanych punktów
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    return filtered_pcd


def remove_floor_ransac(pcd, distance_threshold=0.01):
    """
    Usuwa podłogę z chmury punktów za pomocą segmentacji RANSAC.
    :param pcd: chmura punktów
    :param distance_threshold: próg odległości do klasyfikacji punktów jako pasujących do modelu
    :return: chmura punktów bez podłogi
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000
    )
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return outlier_cloud


# Inicjalizacja kamery RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    point_clouds = []
    start_time = time.time()
    capture_duration = 20  # Czas trwania zbierania danych w sekundach

    while time.time() - start_time < capture_duration:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        pcd = frame_to_point_cloud(depth_frame)  # Funkcja zdefiniowana w Twoim kodzie

        # wycięcie fragmentu chmury punktów
        bounds = {
            "x_min": -1.5,
            "x_max": 1.5,
            "y_min": -1,
            "y_max": 1,
            "z_min": 1,
            "z_max": 2.5,
        }
        filtered_pcd = crop_point_cloud(pcd, bounds)

        # # usunięcie podłogi
        filtered_pcd = remove_floor_ransac(filtered_pcd, distance_threshold=0.03)

      
        # filtracja odstających punktów
        filtered_pcd = statistical_outlier_removal_open3d(
            filtered_pcd, k_neighbors=1000, std_ratio=2
        )

        point_clouds.append(filtered_pcd)

    # Zapis chmury punktów do pliku
    write_point_cloud_to_file(point_clouds)
    # # Rejestracja chmur punktów (Implementacja algorytmu ICP)
    # r_pcd = register_point_clouds(point_clouds)

    # # Rekonstrukcja powierzchni (np. triangulacja Delaunaya, marching cubes)
    # mesh = reconstruct_surface(r_pcd)

    # Obróbka i optymalizacja modelu
    # ...

    # # Wizualizacja i eksport modelu
    # o3d.visualization.draw_geometries([mesh])

finally:
    points = np.asarray(filtered_pcd.points)
    pipeline.stop()

    x = (points[:, 0],)  # współrzędne X
    y = (points[:, 1],)  # współrzędne Y
    z = (points[:, 2],)  # współrzędne Z
    trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(
            size=1,
            color=points[:, 2],  # Kolorowanie punktów w oparciu o ich wartość Z
            colorscale="Viridis",  # Wybór skali kolorów
            showscale=True,  # Pokazuje pasek skali kolorów
        ),
    )

    # Definiowanie układu wykresu
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),  # Marginesy
        scene=dict(
            xaxis=dict(title="X Axis"),  # Tytuł osi X
            yaxis=dict(title="Y Axis"),  # Tytuł osi Y
            zaxis=dict(title="Z Axis"),  # Tytuł osi Z
        ),
    )

    fig = go.Figure(data=[trace], layout=layout)
    fig.show()

    while True:
        pass
