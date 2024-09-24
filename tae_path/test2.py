import time
import numpy as np
from Arm_Lib import Arm_Device

# 링크 길이 설정
L1 = 3.846  # Base에서 Shoulder까지의 길이
L2 = 8.327  # Shoulder에서 Elbow까지의 길이
L3 = 8.327  # Elbow에서 Wrist까지의 길이
L4 = 18.7   # Wrist에서 End-effector까지의 길이

# 역기구학 함수 정의
def inverse_kinematics(x, y, z):
    # XY 평면에서의 거리 r 계산
    r = np.sqrt(x**2 + y**2)
    
    # θ1 계산 (Base에서의 회전 각도)
    theta1 = np.arctan2(y, x)    
    
    # D 값 계산: 로봇 팔이 도달할 수 있는지 여부 확인
    D = (r**2 + z**2 - L2**2 - L3**2) / (2 * L2 * L3)
    print(D)
    if D < -1 or D > 1:
        raise ValueError("해당 위치에 팔이 도달할 수 없습니다.")
    
    # θ3 계산 (Elbow 각도)
    theta3 = np.arctan2(-np.sqrt(1 - D**2), D)  # Elbow 각도 (θ3)
    theta3 += np.radians(90)  # 기준이 90도이므로 추가
    
    # θ2 계산 (Shoulder 각도)
    theta2 = np.arctan2(z_prime, r) - np.arctan2(L3 * np.sin(theta3 - np.radians(90)), L2 + L3 * np.cos(theta3 - np.radians(90)))
    theta2 += np.radians(90)  # 기준이 90도이므로 추가
    
    # θ4 계산: End-effector의 회전 각도
    theta4 = 90 - (theta2 + theta3 - np.radians(180))  # 직선 상태를 기준으로 회전 보정
    
    return np.degrees(theta1), np.degrees(theta2), np.degrees(theta3), np.degrees(theta4)

# 서보 모터에 맞게 각도를 조정하는 함수
def adjust_angle_for_servo(angle):
    if angle < 0:
        return 0
    elif angle > 180:
        return 180
    return angle

# 목표 좌표 설정
x_target = 30
y_target = 30
z_target = 20

# 로봇 팔 객체 생성
Arm = Arm_Device()

# 잠시 대기
time.sleep(0.1)

# 메인 제어 함수
def main():
    # 역기구학을 이용해 각도 계산
    try:
        theta1, theta2, theta3, theta4 = inverse_kinematics(x_target, y_target, z_target)
    except ValueError as e:
        print(e)
        return
    
    # 계산된 각도를 서보 모터에 맞게 조정
    theta1 = adjust_angle_for_servo(theta1)
    theta2 = adjust_angle_for_servo(theta2)
    theta3 = adjust_angle_for_servo(theta3)
    theta4 = adjust_angle_for_servo(theta4)
    
    print(f"Adjusted joint angles: θ1={theta1:.2f}, θ2={theta2:.2f}, θ3={theta3:.2f}, θ4={theta4:.2f}")
    
    # 계산된 각도를 사용하여 로봇 팔 움직임 제어
    Arm.Arm_serial_servo_write(1, int(theta1), 2000)  # Base 서보 제어
    time.sleep(0.5)
    
    Arm.Arm_serial_servo_write(2, int(theta2), 2000)  # Shoulder 서보 제어
    time.sleep(0.5)
    
    Arm.Arm_serial_servo_write(3, int(theta3), 2000)  # Elbow 서보 제어
    time.sleep(0.5)
    
    Arm.Arm_serial_servo_write(4, int(theta4), 2000)  # Wrist 서보 제어 (회전 포함할 경우)
    time.sleep(1)

try:
    main()
except KeyboardInterrupt:
    print("프로그램이 종료되었습니다!")
    pass

# 로봇 팔 객체 해제
del Arm


