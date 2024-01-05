import pyrealsense2 as rs
import open3d as o3d
import numpy as np

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


def remove_floor_ransac(pcd, distance_threshold=0.01, max_iterations=1000):
    """
    Usuwa podłogę z chmury punktów za pomocą segmentacji RANSAC.
    :param pcd: chmura punktów
    :param distance_threshold: próg odległości do klasyfikacji punktów jako pasujących do modelu
    :return: chmura punktów bez podłogi
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=3, num_iterations=max_iterations
    )
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return outlier_cloud
