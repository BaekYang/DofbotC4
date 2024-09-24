#!/usr/bin/env python3
#coding=utf-8

import time
from Arm_Lib import Arm_Device

# 로봇 팔 객체 생성
Arm = Arm_Device()

# 잠시 대기
time.sleep(0.1)

def main():
    Arm.Arm_serial_servo_write6(0, 90, 20, 90, 90, 0, 1500)
    time.sleep(2)
    #Arm.Arm_serial_servo_write(1, 0, 500)
    #time.sleep(1)
    
    #Arm.Arm_serial_servo_write(3,20, 1500)
    #time.sleep(1)
    #Arm.Arm_serial_servo_write(6,0, 1000)
    #time.sleep(1)
    Arm.Arm_serial_servo_write(6, 180, 1000)
    time.sleep(2)
    
try:
    main()
except KeyboardInterrupt:
    print("프로그램이 종료되었습니다!")
    pass

# 로봇 팔 객체 해제
del Arm
