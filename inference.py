import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import os
from params import train_params

def run_inference():
    # Define the relative path to the model file
    weight_path = os.path.join(train_params.project, 'weights', 'best.pt')

    # Load YOLOv8 model using the relative path
    model = YOLO(weight_path)

    # Camera intrinsics (replace with actual calibration data)
    fx, fy = 2227.93, 2211.78  # Focal length (pixel units)
    cx, cy = 1261.84, 2100.96  # Principal point (pixel units)

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
                X = ( ((x_center - cx) * z_center * 100) / fx ) # X in cm
                Y = ( ((y_center - cy) * z_center * 100) / fy ) # Y in cm
                Z = z_center * 100  # Z in cm (convert from meters to cm)

                # Print the (X, Y, Z) coordinates of the center of the bounding box
                print(f'World coordinates: (X: {X:.2f} cm, Y: {Y:.2f} cm, Z: {Z:.2f} cm)')

                # Draw bounding box and label
                cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                
                # Draw class label above the bounding box
                label = f'{model.names[int(class_id)]}: {conf:.2f}'
                label_position = (int(x1), int(y1) - 10)  # Positioned above the bounding box
                cv2.putText(color_image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Draw world coordinates (X, Y, Z) below the bounding box
                coord_label = f'X: {X:.2f} cm, Y: {Y:.2f} cm, Z: {Z:.2f} cm'
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
    
