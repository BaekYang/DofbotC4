import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import os
from params import train_params

# 회전 행렬을 적용하는 함수
def apply_rotation_matrix(X, Y, Z):
    R_total = np.array([
        [-np.cos(np.radians(135)) * np.cos(np.radians(45)), np.sin(np.radians(135)), -np.cos(np.radians(135)) * np.sin(np.radians(45))],
        [np.sin(np.radians(135)) * np.cos(np.radians(45)), np.cos(np.radians(135)), np.sin(np.radians(135)) * np.sin(np.radians(45))],
        [-np.sin(np.radians(45)), 0, np.cos(np.radians(45))]
    ])

    # Original point as a column vector
    point = np.array([X, Y, Z])

    # Apply the rotation matrix
    rotated_point = np.dot(R_total, point)

    return rotated_point[0], rotated_point[1], rotated_point[2]

# 이동 벡터를 적용하는 함수
def apply_translation(X, Y, Z, Tx, Ty, Tz):
    X_translated = X + Tx
    Y_translated = Y + Ty
    Z_translated = Z + Tz
    return X_translated, Y_translated, Z_translated

# 이동 벡터를 좌표마다 유동적으로 계산하는 함수
def calculate_translation(target_X, target_Y, target_Z, X_rot, Y_rot, Z_rot):
    Tx = target_X - X_rot
    Ty = target_Y - Y_rot
    Tz = target_Z - Z_rot
    return Tx, Ty, Tz

def run_inference():
    # Define the relative path to the model file
    weight_path = os.path.join(train_params.project, 'weights', 'best.pt')

    # Load YOLOv8 model using the relative path
    model = YOLO(weight_path)

    # Camera intrinsics (replace with actual calibration data)
    fx, fy = 595.285932, 597.935197  # Focal length (pixel units)
    cx, cy = 320, 240  # Principal point (pixel units)

    # Configure depth and color streams from the depth camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Set confidence threshold for detections
    CONFIDENCE_THRESHOLD = 0.5

    try:
        while True:
            # Get both color and depth frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Perform YOLOv8 detection on the RGB frame
            results = model(color_image)

            # Extract bounding boxes and process detections
            detections = results[0].boxes
            for box in detections:
                conf = box.conf[0]  # Confidence
                if conf < CONFIDENCE_THRESHOLD:
                    continue  # Skip detections below threshold

                x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
                class_id = box.cls[0]  # Class ID
                
                # Calculate the center of the bounding box
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)
                
                # Get depth (z-coordinate) at the center of the detected object
                z_center = depth_frame.get_distance(x_center, y_center)  # Depth in meters

                # Convert (x, y, z) to world coordinates (in cm)
                X = ((x_center - cx) * z_center * 100) / fx  # X in cm
                Y = ((y_center - cy) * z_center * 100) / fy  # Y in cm
                Z = z_center * 100  # Z in cm (convert from meters to cm)

                # 회전 행렬 적용
                X_rot, Y_rot, Z_rot = apply_rotation_matrix(X, Y, Z)

                # 목표 좌표 설정 (예: (27.3, 23, 0))
                target_X, target_Y, target_Z = 27.3, 23, 0

                # 이동 벡터 유동적으로 계산
                Tx, Ty, Tz = calculate_translation(target_X, target_Y, target_Z, X_rot, Y_rot, Z_rot)

                # 이동 벡터 적용 후 최종 좌표 계산
                X_final, Y_final, Z_final = apply_translation(X_rot, Y_rot, Z_rot, Tx, Ty, Tz)

                # Print the original, rotated, and final world coordinates
                print(f'Original World coordinates: (X: {X:.2f} cm, Y: {Y:.2f} cm, Z: {Z:.2f} cm)')
                print(f'Rotated World coordinates: (X_rot: {X_rot:.2f} cm, Y_rot: {Y_rot:.2f} cm, Z_rot: {Z_rot:.2f} cm)')
                print(f'Final World coordinates: (X_final: {X_final:.2f} cm, Y_final: {Y_final:.2f} cm, Z_final: {Z_final:.2f} cm)')

                # Draw bounding box and label
                cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                
                # Draw class label above the bounding box
                label = f'{model.names[int(class_id)]}: {conf:.2f}'
                label_position = (int(x1), int(y1) - 10)  # Positioned above the bounding box
                cv2.putText(color_image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Draw final world coordinates (X, Y, Z) below the bounding box
                coord_label = f'X: {X_final:.2f} cm, Y: {Y_final:.2f} cm, Z: {Z_final:.2f} cm'
                coord_label_position = (int(x1), int(y2) + 20)  # Positioned below the bounding box
                cv2.putText(color_image, coord_label, coord_label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Display the RGB frame with detections
            cv2.imshow('Real-Time Detection with 3D Coordinates', color_image)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop the pipeline
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_inference()



