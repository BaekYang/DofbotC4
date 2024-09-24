from Arm_Lib import Arm_Device
import time

Arm = Arm_Device()

time.sleep(0.1)

Arm.Arm_serial_servo_write(6,90,1000)
time.sleep(1)