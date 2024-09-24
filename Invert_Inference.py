import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import os
from params import train_params

# 동차 변환 행렬 적용 함수
def apply_homogeneous_transformation(X, Y, Z):
    # 주어진 변환 행렬 T
    T = np.array([
        [-1, 0, 0, 26.8],
        [0, 0.707, -0.707, 69.5],
        [0, -0.707, -0.707, 38.8],
        [0, 0, 0, 1]
    ])

    # 입력 좌표를 동차 좌표계로 확장 (X, Y, Z, 1)
    point = np.array([X, Y, Z, 1])

    # 변환 행렬을 좌표에 적용
    transformed_point = np.dot(T, point)

    # 변환된 좌표 반환
    return transformed_point[0], transformed_point[1], transformed_point[2]

# 교차점 계산 함수
def line_intersection(x1, y1, x3, y3, x2, y2, x4, y4):
    denom = (x1 - x3) * (y2 - y4) - (y1 - y3) * (x2 - x4)
    if denom == 0:
        return None  # 직선이 평행하여 교차점이 없음
    
    Px = ((x1 * y3 - y1 * x3) * (x2 - x4) - (x1 - x3) * (x2 * y4 - y2 * x4)) / denom
    Py = ((x1 * y3 - y1 * x3) * (y2 - y4) - (y1 - y3) * (x2 * y4 - y2 * x4)) / denom
    
    return int(Px), int(Py)

def run_inference():
    # 모델 파일 경로 설정
    weight_path = os.path.join(train_params.project, 'weights', 'best.pt')

    # YOLOv8 모델 로드
    model = YOLO(weight_path)

    # 카메라 내부 파라미터 (실제 보정 데이터를 사용해야 함)
    fx, fy = 595.29, 597.94  # 초점 거리 (픽셀 단위)
    cx, cy = 320, 240  # 주점 (픽셀 단위)

    # 깊이 카메라에서 컬러 및 깊이 스트림 설정
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 스트리밍 시작
    pipeline.start(config)

    CONFIDENCE_THRESHOLD = 0.5

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            results = model(color_image)

            detections = results[0].boxes
            for box in detections:
                conf = box.conf[0]
                if conf < CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

                class_id = int(box.cls[0])
                label = f'{model.names[class_id]} {conf:.2f}'
                cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if hasattr(results[0], 'keypoints'):
                    keypoints = results[0].keypoints[0]

                    if len(keypoints.xy.cpu().numpy()[0]) >= 4:
                        x1_kp, y1_kp = int(keypoints.xy.cpu().numpy()[0][0][0]), int(keypoints.xy.cpu().numpy()[0][0][1])
                        x2_kp, y2_kp = int(keypoints.xy.cpu().numpy()[0][1][0]), int(keypoints.xy.cpu().numpy()[0][1][1])
                        x3_kp, y3_kp = int(keypoints.xy.cpu().numpy()[0][2][0]), int(keypoints.xy.cpu().numpy()[0][2][1])
                        x4_kp, y4_kp = int(keypoints.xy.cpu().numpy()[0][3][0]), int(keypoints.xy.cpu().numpy()[0][3][1])

                        intersection_point = line_intersection(x1_kp, y1_kp, x3_kp, y3_kp, x2_kp, y2_kp, x4_kp, y4_kp)

                        if intersection_point:
                            x_center, y_center = intersection_point

                            z_center = depth_frame.get_distance(x_center, y_center)

                            X = ((x_center - cx) * z_center * 100) / fx
                            Y = ((y_center - cy) * z_center * 100) / fy
                            Z = z_center * 100

                            # 원본 좌표 출력 (화면 및 콘솔)
                            coord_label = f' X={X:.2f} cm, Y={Y:.2f} cm, Z={Z:.2f} cm'
                            cv2.putText(color_image, coord_label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            print(f'원본 좌표: X={X:.2f} cm, Y={Y:.2f} cm, Z={Z:.2f} cm')

                            # 동차 변환 행렬을 적용하여 변환된 좌표 계산
                            X_trans, Y_trans, Z_trans = apply_homogeneous_transformation(X, Y, Z)

                            # 변환된 좌표는 콘솔에만 출력
                            print(f'변환된 좌표: X={X_trans:.2f} cm, Y={Y_trans:.2f} cm, Z={Z_trans:.2f} cm')

                            # 원본 키포인트를 화면에 그림
                            for (x_kp, y_kp) in [(x1_kp, y1_kp), (x2_kp, y2_kp), (x3_kp, y3_kp), (x4_kp, y4_kp)]:
                                cv2.circle(color_image, (x_kp, y_kp), 5, (0, 255, 0), -1)
                            cv2.circle(color_image, (x_center, y_center), 4, (255, 0, 0), -1)

            cv2.imshow('Real-Time Detection with 3D Coordinates', color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_inference()







