#!/usr/bin/env python3
#coding=utf-8

import time
import math
import numpy as np
from Arm_Lib import Arm_Device

# 로봇 팔 객체 생성
Arm = Arm_Device()

# 잠시 대기
time.sleep(0.1)

# 역기구학 함수 (이전 코드에서 제공된 것)
L2 = 8.327  # Shoulder에서 Elbow까지의 길이
L3 = 8.327  # Elbow에서 Wrist까지의 길이
L4 = 18.7   # Wrist에서 End-effector까지의 길이

def inverse_kinematics(x, y, z):
    # Wrist 위치 계산
    x_prime = x
    y_prime = y
    z_prime = z - L4  # End-effector 길이 조정
    
    # Base 회전 각도 계산 (θ1)
    theta1 = math.atan2(y_prime, x_prime)
    
    # 평면 거리 계산
    r = math.sqrt(x_prime**2 + y_prime**2)
    
    # D 계산
    D = (r**2 + z_prime**2 - L2**2 - L3**2) / (2 * L2 * L3)
    
    if D > 1:
        D = 1
    elif D < -1:
        D = -1
    
    theta3 = math.atan2(-math.sqrt(1 - D**2), D)  # Elbow 각도 (θ3)
    theta2 = math.atan2(z_prime, r) - math.atan2(L3 * math.sin(theta3), L2 + L3 * math.cos(theta3))  # Shoulder 각도 (θ2)
    
    theta4 = 0  # Wrist의 회전 각도는 따로 필요에 따라 추가 가능
    
    # 각도를 도(degrees)로 변환하여 반환
    return np.degrees(theta1), np.degrees(theta2), np.degrees(theta3), np.degrees(theta4)

# 각도를 서보 모터에 맞게 조정하는 함수
def adjust_angle_for_servo(angle):
    """
    서보 모터가 음수 각도를 처리할 수 없으므로, 음수 각도를 0도로 변환하고,
    0~180도 범위로 제한합니다.
    """
    if angle < 0:
        return 0
    elif angle > 180:
        return 180
    return angle

# 목표 좌표 설정
x_target = 30
y_target = 4
z_target = 5

# 메인 제어 함수
def main():
    # 서보를 중앙 위치로 초기화


    # 역기구학을 이용해 각도 계산
    theta1, theta2, theta3, theta4 = inverse_kinematics(x_target, y_target, z_target)
    
    # 계산된 각도를 서보 모터에 맞게 조정
    theta1 = adjust_angle_for_servo(theta1)
    theta2 = adjust_angle_for_servo(theta2)
    theta3 = adjust_angle_for_servo(theta3)
    theta4 = adjust_angle_for_servo(theta4)
    
    print(f"Adjusted joint angles: θ1={theta1:.2f}, θ2={theta2:.2f}, θ3={theta3:.2f}, θ4={theta4:.2f}")
    
    # 계산된 각도를 사용하여 로봇 팔 움직임 제어
    Arm.Arm_serial_servo_write(1, int(theta1), 2000)  # Base 서보 제어
    time.sleep(0.5)
    
    #Arm.Arm_serial_servo_write(2, int(theta2), 2000)  # Shoulder 서보 제어
    #time.sleep(2)
    
    #Arm.Arm_serial_servo_write(3, int(theta3), 2000)  # Elbow 서보 제어
    #time.sleep(0.5)
    
    #Arm.Arm_serial_servo_write(4, int(theta4), 2000)  # Wrist 서보 제어 (회전 포함할 경우)
    #time.sleep(1)


try:
    main()
except KeyboardInterrupt:
    print("프로그램이 종료되었습니다!")
    pass

# 로봇 팔 객체 해제
del Arm

