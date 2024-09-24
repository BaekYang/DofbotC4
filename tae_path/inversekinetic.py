import numpy as np
import math

# 로봇 팔의 각 링크 길이 설정 (L4는 Wrist에서 End-effector까지의 길이)
L2 = 8.327  # Shoulder에서 Elbow까지의 길이
L3 = 8.327  # Elbow에서 Wrist까지의 길이
L4 = 18.7   # 실제로 측정한 Wrist에서 End-effector까지의 길이

def inverse_kinematics(x, y, z):
    """
    주어진 (x, y, z) 좌표에 도달하기 위한 각 관절의 각도를 계산합니다.
    """
    
    # Wrist의 위치를 계산 (목표 지점에서 End-effector의 길이를 뺌)
    x_prime = x
    y_prime = y
    z_prime = z - L4  # z 축에서 L4만큼 조정
    
    # θ1: Base 회전 (회전각도는 x, y 평면에서의 위치에 따라 결정)
    theta1 = math.atan2(y_prime, x_prime)

    # r: x_prime과 y_prime의 평면 거리
    r = math.sqrt(x_prime**2 + y_prime**2)
    
    # Wrist 위치에서 r과 z_prime을 이용한 θ2, θ3 계산
    D = (r**2 + z_prime**2 - L2**2 - L3**2) / (2 * L2 * L3)
    print(D)
    # D 값이 범위를 벗어나면 팔이 해당 위치에 도달할 수 없음을 의미
    if D > 1:
        D = 1  # D 값을 1로 고정
    elif D < -1:
        D = -1  # D 값을 -1로 고정
    
    theta3 = math.atan2(-math.sqrt(1 - D**2), D)  # Elbow 각도
    theta2 = math.atan2(z_prime, r) - math.atan2(L3 * math.sin(theta3), L2 + L3 * math.cos(theta3))  # Shoulder 각도
    
    # θ4: Wrist의 각도는 End-effector 방향을 제어 (필요에 따라 추가)
    theta4 = 0  # 고정된 각도로 할당하거나, 특정 방향을 원할 경우 계산
    
    return np.degrees(theta1), np.degrees(theta2), np.degrees(theta3), np.degrees(theta4)

# 목표 위치 설정 (예: x=15, y=10, z=20)
x_target = 5
y_target = 5
z_target = 5

try:
    angles = inverse_kinematics(x_target, y_target, z_target)
    print(f"Calculated joint angles: {angles}")
except ValueError as e:
    print(e)




