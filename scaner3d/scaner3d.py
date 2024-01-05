import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import time
import plotly.graph_objs as go
import filters 
import computing 
import merging 

def show_pcd_open3d(points): 
    x=points[:, 0],  # współrzędne X
    y=points[:, 1],  # współrzędne Y
    z=points[:, 2],  # współrzędne Z
    trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=1,
            color=points[:, 2],  # Kolorowanie punktów w oparciu o ich wartość Z
            colorscale='Viridis',  # Wybór skali kolorów
            showscale=True         # Pokazuje pasek skali kolorów
        )
    )

    # Definiowanie układu wykresu
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),  # Marginesy
        scene=dict(
            xaxis=dict(title='X Axis'),    # Tytuł osi X
            yaxis=dict(title='Y Axis'),    # Tytuł osi Y
            zaxis=dict(title='Z Axis')     # Tytuł osi Z
        )
    )

    fig = go.Figure(data=[trace], layout=layout)
    fig.show()

# Inicjalizacja kamery RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)





def start_recording(pipeline ):
    def collect_data(num_points):
        collected_points = []
        start_time = time.time()
        while len(collected_points) < num_points:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            pcd = computing.frame_to_point_cloud(depth_frame, frames)

            bounds = {
            "x_min": -1,
            "x_max": 1,
            "y_min": -1,
            "y_max": 1,
            "z_min": 1,
            "z_max": 2.5,
               }
            
            pcd = filters.crop_point_cloud(pcd, bounds)
            pcd = filters.remove_floor_ransac(pcd, distance_threshold=0.03, max_iterations=1000)    
            pcd = filters.statistical_outlier_removal_open3d(pcd, 123, 1.0)
            
            # dalsze przetwarzanie pcd...
            collected_points.append(pcd)

        return collected_points

    print("zapraszam do skanowania")
    input()
    print("3" )
    time.sleep(1)
    print("2" )
    time.sleep(1)
    print("1" )
    time.sleep(1)
    print("skan!!!!" )
    # Pierwsza partia zbierania danych
    point_clouds_first_part = collect_data(3)
    computing.write_point_cloud_to_file(point_clouds_first_part)

    # Wyświetlenie komunikatu o obróceniu się
    print("Obróć się o 180 stopni i naciśnij Enter, aby kontynuować...")
    input()
    print("3" )
    time.sleep(1)
    print("2" )
    time.sleep(1)
    print("1" )
    time.sleep(1)
    print("skan!!!!" )

    # Druga partia zbierania danych
    point_clouds_second_part = collect_data(3)
    computing.write_point_cloud_to_file(point_clouds_second_part)

    show_pcd_open3d(np.asarray(point_clouds_first_part[0].points))  
    show_pcd_open3d(np.asarray(point_clouds_second_part[0].points))

    # Tutaj możesz wykonać dalsze operacje na zebranych danych
                                 

start_recording(pipeline)
pipeline.stop()