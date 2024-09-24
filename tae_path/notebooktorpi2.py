import socket
import pickle
import struct

# 소켓 설정
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("서버에 연결 시도 중...")

try:
    # 노트북의 IP 주소와 포트 번호를 사용하여 연결
    client_socket.connect(('192.168.0.32', 8485))
    print("서버에 성공적으로 연결됨!")
except Exception as e:
    print(f"서버 연결 실패: {e}")
    exit()

data = b""
payload_size = struct.calcsize("L")  # 메시지 크기를 수신할 크기 계산
print("데이터 수신 대기 중...")

try:
    while True:
        # 메시지 크기 수신
        print("메시지 크기 수신 중...")
        while len(data) < payload_size:
            packet = client_socket.recv(4096)
            if not packet:
                print("데이터 패킷을 받을 수 없습니다. 연결이 끊어졌을 수 있습니다.")
                break
            data += packet  # 패킷 추가

        # 수신된 패킷에서 메시지 크기 확인
        if len(data) >= payload_size:
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("L", packed_msg_size)[0]
            print(f"수신한 메시지 크기: {msg_size} 바이트")
        else:
            print("메시지 크기를 수신하지 못했습니다.")
            break

        # 실제 XYZ 데이터 수신
        print("XYZ 데이터 수신 중...")
        while len(data) < msg_size:
            packet = client_socket.recv(4096)
            if not packet:
                print("XYZ 데이터 수신 중 연결 끊김")
                break
            data += packet

        if len(data) >= msg_size:
            xyz_data = data[:msg_size]
            data = data[msg_size:]

            # XYZ 좌표 역직렬화
            x, y, z = pickle.loads(xyz_data)
            print(f"수신한 좌표: (x: {x}, y: {y}, z: {z} 미터)")

except Exception as e:
    print(f"오류 발생: {e}")

finally:
    print("클라이언트 소켓 종료")
    client_socket.close()


