#!/usr/bin/env python3
#coding=utf-8

import time
from Arm_Lib import Arm_Device

# 로봇 팔 객체 생성
Arm = Arm_Device()

# 잠시 대기
time.sleep(0.1)

def main():
    Arm.Arm_serial_servo_write6(167, 114, 35, 52, 180, 170, 1500)
    time.sleep(1)
    Arm.Arm_serial_servo_write6(138, 60, 64, 60, 180, 172, 1500)
    time.sleep(1)
    # 서보를 중앙 위치로 초기화
    #Arm.Arm_serial_servo_write(2, 60, 1000)
    #time.sleep(1)
    #Arm.Arm_serial_servo_write(1, 130, 1000)
    #time.sleep(0.5)
    #Arm.Arm_serial_servo_write(2, 52, 1000)
    #time.sleep(1)
    #Arm.Arm_serial_servo_write(3, 17, 500)
    #time.sleep(1)
    #Arm.Arm_serial_servo_write(2, 50, 1500)
    #time.sleep(1)
    
try:
    main()
except KeyboardInterrupt:
    print("프로그램이 종료되었습니다!")
    pass

# 로봇 팔 객체 해제
del Arm