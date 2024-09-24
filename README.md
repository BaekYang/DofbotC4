# DofbotC4

[학습] 
1. 준비 :
data 폴더를 만들고 그 안에 train, val, test 폴더를 형성한다.
그리고 그 밑에 image폴더를 만든다.
image폴더 안에는 png나 jpg파일, 그리고 그 파일에 대응하는 json파일을 넣는다.
이때 확장명 빼고 파일이름은 꼭 일치시킨다.

2. 얌파일 만들기 :
data.py를 돌려서 json을 yaml로 바꾼다.

3. 레이블 형성 :
data 폴더 안의 train, val, test 폴더 하위로 label 폴더를 만든다.
yolo로 읽을 txt파일들이다.

4. 학습시작
train.py 를 시작한다.

5. pt파일을 기반으로 Inference를 돌려 좌표에 대한 json파일을 얻는다.

깔아야하는거!

[로봇암 코드]
tae : 
juicy :

[로봇암 디텍션 코드]
tae :
juicy :

cuda 12.1에 맞는 Pytorch, numpy, 싸이킷런, ultralytics 등등.. gpt한테 cuda 12.1에 맞게 알려달라고 하세영! 

realsense 카메라 설치 (인텔 realsense2 - 라즈베리파이 4에 깔기)

sudo apt-get update
sudo apt-get install git cmake g++ libgtk-3-dev libglib2.0-dev
sudo apt-get install libboost-all-dev
sudo apt-get install libssl-dev

git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense

mkdir build && cd build
cmake ../ -DBUILD_EXAMPLES=true -DBUILD_GRAPHICAL_EXAMPLES=false
make -j$(nproc)
sudo make install

sudo ldconfig

realsense-viewer

