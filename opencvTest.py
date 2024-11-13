import cv2
import time

# 웹캠 시작
#cap = cv2.VideoCapture(1)
#cap = cv2.VideoCapture("/dev/video1", cv2.CAP_FFMPEG)
cap = cv2.VideoCapture("/dev/video1", cv2.CAP_V4L2)

# 웹캠 설정

# MJPG 형식으로 설정
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# 이미지 크기 설정 (예: 640x480 -> fhd:1920x1080(30fps) -> hd:1280*720(60fps))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)  # 60fps로 설정


# 웹캠이 열렸는지 확인
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

frame_count = 0 # 1초동안 웹캠으로부터 읽은 프레임의 수
start_time = time.time()

try:
    while True:
        # 웹캠에서 한 프레임 읽기
        ret, frame = cap.read()

        # 성공적으로 프레임을 읽었는지 확인
        if ret:
            frame_count += 1

            # 현재 시간
            current_time = time.time()

            # 지난 시간 계산
            elapsed_time = current_time - start_time

            # 지난시간이 1초가 되면
            if elapsed_time >= 1:
                # 프레임 속도 계산 및 출력면
                fps = frame_count / elapsed_time
                print("FPS: {:.2f}".format(fps))

                # 카운터 및 타이머 초기화
                frame_count = 0
                start_time = time.time()

            # 프레임 표시 -> 모델 추정으로 수정할 부분!
            cv2.imshow('Webcam Frame', frame)

            # 키 입력 대기 (예: 'q'를 누르면 중단)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("프레임을 읽을 수 없습니다.")
            break
except KeyboardInterrupt:
    # Ctrl+C를 누르면 종료
    pass

print("내장 메소드 사용")
print(cap.get(cv2.CAP_PROP_FPS))

# 사용이 끝났으면 웹캠을 해제
cap.release()
cv2.destroyAllWindows()

