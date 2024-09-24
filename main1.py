import cv2
import time
from ultralytics import YOLO
from Arm_Lib import Arm_Device

# 로봇 팔 객체 생성
Arm = Arm_Device()

# YOLOv8 모델 로드 (juicy1.pt 파일 경로)
model = YOLO('/home/juicy/Juicy_code/best.pt')

# 카메라 설정 (depth camera index, 기본값 0)
cap = cv2.VideoCapture("/dev/video4")
time.sleep(2)  # 카메라 워밍업 시간

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

time.sleep(2)  # 카메라 워밍업 시간

# 로봇 팔 동작 함수 (컵을 잡는 동작)
def catch():
    print("컵이 올바르게 위치했습니다. 로봇 팔이 동작을 시작합니다.")
    Arm.Arm_serial_servo_write6(180, 30, 0, 165, 180, 180, 2000)
    time.sleep(2)
    Arm.Arm_serial_servo_write(6, 140, 1500)
    time.sleep(1)
    Arm.Arm_serial_servo_write6(180, 23, 20, 150, 180, 120, 1500)
    time.sleep(1)
    Arm.Arm_serial_servo_write(6, 180, 1000)
    time.sleep(1)

# 로봇 팔 이동 함수
def move():
    Arm.Arm_serial_servo_write6(167, 114, 35, 52, 180, 180, 1500)
    time.sleep(1)
    Arm.Arm_serial_servo_write6(141, 70, 64, 60, 180, 180, 1300)
    time.sleep(1)

# 물 따르는 동작 함수
def water():
    Arm.Arm_serial_servo_write(5, 90, 1500)
    time.sleep(5)
    Arm.Arm_serial_servo_write(5, 180, 1500)
    time.sleep(1)

# 로봇 팔 원래 위치로 복귀 함수
def return_move():
    Arm.Arm_serial_servo_write6(180, 114, 35, 52, 180, 180, 1500)
    time.sleep(1)   
    Arm.Arm_serial_servo_write6(180, 23, 20, 150, 180, 180, 2000)
    time.sleep(2)
    Arm.Arm_serial_servo_write(6, 152, 1000)
    time.sleep(1)
    Arm.Arm_serial_servo_write6(180, 24, 8, 165, 180, 152, 1000)
    time.sleep(1)
    Arm.Arm_serial_servo_write6(180, 24, 6, 162, 180, 143, 1000)
    time.sleep(1)
    Arm.Arm_serial_servo_write6(180, 26, 5, 170, 180, 152, 1000)
    time.sleep(1)
    Arm.Arm_serial_servo_write6(180, 24, 6, 175, 180, 152, 1000)
    time.sleep(7)

# 로봇 팔 초기 위치 함수
def initial():
    Arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 160, 1500)
    time.sleep(1)

# 객체 검출 함수 (YOLOv8 사용)
def detect_object(frame):
    results = model(frame)  # YOLO 모델로 프레임 처리
    boxes = results[0].boxes.xyxy  # 바운딩 박스 좌표
    labels = results[0].boxes.cls  # 클래스 레이블 (숫자)
    confidences = results[0].boxes.conf  # 신뢰도 점수
    return labels, boxes, confidences

# 객체 상태 초기화 (0: 컵 없음, 1: 컵이 정상, 2: 컵이 뒤집힘)
object_state = 0

while True:
    # 카메라에서 실시간 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 프레임을 가져올 수 없습니다.")
        break

    # YOLO 모델을 사용해 객체 검출
    labels, boxes, confidences = detect_object(frame)

    # 프레임 크기 조정 (필요시)
    resized_frame = cv2.resize(frame, (640, 480))

    # 객체 상태 확인 (컵이 정상인지 뒤집혔는지)
    for label in labels:
        if label == 1:  # 클래스 ID 1: 컵이 정상 상태
            object_state = 1
        elif label == 2:  # 클래스 ID 2: 컵이 뒤집힘
            object_state = 2

    # 바운딩 박스와 신뢰도 및 상태 표시
    for box, confidence in zip(boxes, confidences):
        x1, y1, x2, y2 = map(int, box)  # 바운딩 박스 좌표를 정수로 변환
        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 초록색 바운딩 박스 그리기

        # 신뢰도 점수 및 현재 상태를 바운딩 박스 오른쪽 상단에 표시
        conf_text = f'{object_state}                                    {confidence:.2f}'  # 신뢰도와 상태 출력
        cv2.putText(resized_frame, conf_text, (x2-400, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    # 결과를 화면에 표시
    cv2.imshow('Detection', resized_frame)

    # 객체 상태에 따라 동작 수행
    if object_state == 1:
        print("컵이 올바르게 위치했습니다. 잡기 동작을 시작합니다.")
        catch()  # 컵 잡기 동작 함수 호출
        move()
        water()
        return_move()
        initial()
        break  # 한 번 동작 후 종료

    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()

del Arm
