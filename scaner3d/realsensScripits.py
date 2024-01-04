import pyrealsense2 as rs
import numpy as np
import open3d as o3d


def statistical_outlier_removal_open3d(pcd, k_neighbors=20, std_ratio=1.0):
    # Używanie wbudowanej metody Open3D do usuwania odstających punktów
    # Wartości nb_neighbors i std_ratio są używane do określenia, które punkty są odstające
    # Metoda zwraca dwa obiekty PointCloud: jeden z odfiltrowanymi punktami, drugi z usuniętymi punktami
    filtered_pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=k_neighbors, std_ratio=std_ratio
    )
    return filtered_pcd


# def statistical_outlier_removal(pcd, k_neighbors=20, std_ratio=1.0):
#     # Konwersja chmury punktów PointCloud na tablicę numpy
#     points = np.asarray(pcd.points)

#     # Obliczanie odległości między wszystkimi punktami
#     distances = np.sqrt(((points - points[:, np.newaxis])**2).sum(axis=2))

#     # Obliczanie średnich odległości dla k najbliższych sąsiadów
#     sorted_distances = np.sort(distances, axis=1)
#     mean_distances = np.mean(sorted_distances[:, 1:k_neighbors+1], axis=1)

#     # Określenie progu dla odstających punktów
#     mean_dist_global = np.mean(mean_distances)
#     threshold = mean_dist_global + std_ratio * np.std(mean_distances)

#     # Filtracja punktów, które są poniżej progu
#     filtered_indices = mean_distances < threshold
#     filtered_points = points[filtered_indices]

#     # Tworzenie nowego obiektu PointCloud z odfiltrowanymi punktami
#     filtered_pcd = o3d.geometry.PointCloud()
#     filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

#     return filtered_pcd


def radius_outlier_removal_open3d(pcd, radius=0.5, min_neighbors=2):
    # Używanie wbudowanej metody Open3D do usuwania odstających punktów na podstawie promienia
    # Metoda zwraca dwa obiekty PointCloud: jeden z odfiltrowanymi punktami, drugi z usuniętymi punktami
    filtered_pcd, _ = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    return filtered_pcd


# Funkcja do przetwarzania klatki i konwersji na chmurę punktów
def frame_to_point_cloud(depth_frame):
    pc = rs.pointcloud()
    print(type(pc))
    pc.map_to(frames.get_color_frame())
    points = pc.calculate(depth_frame)
    vtx = np.asanyarray(points.get_vertices())
    vtx_np = np.array([(v[0], v[1], v[2]) for v in vtx], dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vtx_np)
    return pcd


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


# Konfiguracja strumienia
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

try:
    # Pobranie klatek
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    pcd = frame_to_point_cloud(depth_frame)
    print(type(pcd))

    # # Lista do przechowywania chmur punktów
    # point_clouds = []

    # for _ in range(3):  # Przetwarzanie 3 klatek
    #     frames = pipeline.wait_for_frames()
    #     depth_frame = frames.get_depth_frame()
    #     color_frame = frames.get_color_frame()

    #     if not depth_frame or not color_frame:
    #         continue

    #     # Dodanie chmury punktów do listy
    #     point_clouds.append(frame_to_point_cloud(depth_frame))

    # # Połączenie chmur punktów
    # combined_pcd = point_clouds[0]
    # for pcd in point_clouds[1:]:
    #     combined_pcd += pcd

    min_dist = 0.5  # minimalna odległość w metrach
    max_dist = 3  # maksymalna odległość w metrach

    filtered_pcd = filter_point_cloud_by_distance(pcd, min_dist, max_dist)
    # Możesz teraz użyć filtered_pcd do dalszego przetwarzania lub wyświetlenia

    filtered_pcd = statistical_outlier_removal_open3d(
        filtered_pcd, k_neighbors=20, std_ratio=1.0
    )
    # filtered_pcd = radius_outlier_removal_open3d(filtered_pcd, radius=0.5, min_neighbors=2)

    # Opcjonalnie: Możesz zastosować dalsze przetwarzanie, np. downsampling
    # combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.005)

    # Wyświetlenie połączonej chmury punktów
    o3d.visualization.draw_geometries([filtered_pcd])
finally:
    pipeline.stop()
