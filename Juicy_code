import cv2
import time
from ultralytics import YOLO  # Import YOLO from ultralytics package

# Load YOLOv8 model (best.pt file)
model = YOLO('/home/juicy/Juicy_code/best.pt')

# Camera setup (depth camera index, default is /dev/video4)
cap = cv2.VideoCapture("/dev/video4")
time.sleep(2)  # Allow time for the camera to warm up

# Function to pour water (function to control the juicy robot arm)
def pour_water():
    print("The robot arm is pouring water.")
    # Example: Add code to set the angles or control the movements of the robot arm
    # motor.move_to_position(...)

# Object detection function (detect objects using YOLOv8)
def detect_object(frame):
    resized_frame = cv2.resize(frame, (640, 640))  # Resize to YOLOv8's default size (640x640)
    results = model(resized_frame)  # Process the frame using the YOLO model
    # Extract object class and bounding boxes from the results
    labels, boxes = results[0].boxes.cls.cpu().numpy(), results[0].boxes.xyxy.cpu().numpy()
    return labels, boxes, resized_frame

# Initialize object state (0: cup is upside down, 1: cup is upright)
object_state = 0  

while True:
    # Read real-time frames from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to retrieve frame from the camera.")
        break

    # Detect objects using the YOLO model
    labels, boxes, resized_frame = detect_object(frame)

    # Draw bounding boxes around detected objects
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)  # Convert bounding box coordinates to integers
        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box in green
    
    # If the cup (class ID 1) is detected
    if 1 in labels:
        object_state = 1

    # If object state becomes 1, execute the pouring action
    if object_state == 1:
        print("The cup is correctly positioned. Starting the pouring action.")
        pour_water()  # Call the function to pour water
        break  # Exit the loop after pouring once

    # Display the result
    cv2.imshow('Detection', resized_frame)  # Show the resized frame with bounding boxes
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

