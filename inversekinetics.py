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

# 링크 길이 설정 (그림에 따른 링크 길이)
L1 = 3.846  # Base에서 Shoulder까지의 길이
L2 = 8.327  # Shoulder에서 Elbow까지의 길이
L3 = 8.327  # Elbow에서 Wrist까지의 길이
L4 = 18.7   # Wrist에서 End-effector까지의 길이

def inverse_kinematics(x, y, z):
    # phi는 0으로 고정
    phi = 0
    
    # θ₁ 계산 (Base 회전 각도)
    theta1 = math.atan2(y, x)
    
    # r 계산 (xy 평면에서의 거리)
    r = math.sqrt(x**2 + y**2)
    
    # θ₃ 계산 (Elbow 각도)
    cos_theta3 = ( r**2 + (z-L4)**2 - L2**2 - L3**2) / (2 * L2 * L3)
    sin_theta3 = math.sqrt(1 - cos_theta3**2)
    theta3 = math.atan2(sin_theta3, cos_theta3) +np.radians(90)
    
    # θ₂ 계산 (Shoulder 각도)
    K1 = L2 + L3 * math.cos(cos_theta3)
    K2 = L3 * math.sin(sin_theta3)
    rn = r - L4*math.cos(phi)
    zn = z - L4*math.sin(phi)
    theta2 = math.atan2(K1 * zn - K2 * rn, K1 * rn - K2 * zn)

    # θ₄ 계산 (whrist의 각도)
    theta4 = phi - (theta2 + theta3)
    
    return np.degrees(theta1), np.degrees(theta2), np.degrees(theta3), np.degrees(theta4)

# 테스트할 목표 좌표 설정 (x, y, z)
x_target = 10
y_target = 10
z_target = 20

# 각도 계산
angles = inverse_kinematics(x_target, y_target, z_target)
print(f"계산된 각도: θ₁={angles[0]:.2f}, θ₂={angles[1]:.2f}, θ₃={angles[2]:.2f}, θ₄={angles[3]:.2f}")

# 메인 제어 함수
def main():
    # 서보를 중앙 위치로 초기화
    Arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 500)
    time.sleep(1)

    # 역기구학을 이용해 각도 계산
    theta1, theta2, theta3, theta4 = inverse_kinematics(x_target, y_target, z_target)
     
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
