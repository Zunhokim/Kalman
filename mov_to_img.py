import cv2
import os

# MP4 파일 경로와 출력 폴더 경로 설정
video_path = "/Users/junho_kim/Desktop/zunobono/2024_2학기/캡디/Test_set/10_shape.mp4"  # 변환할 MP4 파일 경로
output_folder = "/Users/junho_kim/Desktop/zunobono/2024_2학기/캡디/Test_imgset/10_shape_crop/"  # 프레임 저장 폴더

# 출력 폴더 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# 비디오 정보 확인
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 원본 프레임 너비
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 원본 프레임 높이
print(f"원본 해상도: {frame_width}x{frame_height}")
print(f"비디오 프레임 속도: {fps} FPS")

# 자르기 좌표 계산 (3840x2160 -> 2160x2160 가운데 부분)
crop_size = 2160
start_x = (frame_width - crop_size) // 2  # 가로 중앙에서 시작
start_y = 0  # 세로는 위에서부터 시작
end_x = start_x + crop_size
end_y = start_y + crop_size

# 프레임 저장
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:  # 더 이상 프레임이 없으면 루프 종료
        break
    
    # 프레임 자르기 (3840x2160 -> 2160x2160)
    cropped_frame = frame[start_y:end_y, start_x:end_x]

    frame_filename = os.path.join(output_folder, f"10_shape_crop_{frame_count:04d}.png")
    cv2.imwrite(frame_filename, cropped_frame)  # 프레임 저장
    print(f"{frame_filename} save complete.")
    frame_count += 1

# 자원 해제
cap.release()
print(f"총 {frame_count}개의 프레임이 저장되었습니다.")
