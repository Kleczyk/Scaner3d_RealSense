import pyrealsense2 as rs

# Konfiguracja strumienia
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.accel)


pipeline.start(config)


try:
    while True:
        # Pobieranie zestawu klatek
        frames = pipeline.wait_for_frames()

        # Pobieranie klatki akcelerometru
        accel_frame = frames.first_or_default(rs.stream.accel)
        if accel_frame:
            # Konwersja na obiekt danych ruchu
            accel_data = accel_frame.as_motion_frame().get_motion_data()

            # Wydrukowanie danych akcelerometru
            print("Accel data: x = {0:.3f}, y = {1:.3f}, z = {2:.3f}".format(accel_data.x, accel_data.y, accel_data.z))

finally:
    # Zatrzymanie strumienia
    pipeline.stop()
