import cv2
import glob
import numpy as np
import os

# Parameters
image_folder = '/Users/junho_kim/Downloads/crossline_image'  # 이미지가 있는 폴더 경로
output_folder = '/Users/junho_kim/Downloads/crossline_image_green'  # 결과 파일 저장 폴더
green_lower = np.array([40, 50, 50])  # 초록색 하한값 (HSV)
green_upper = np.array([80, 255, 255])  # 초록색 상한값 (HSV)

# 이미지 파일 가져오기
images = sorted(glob.glob(f"{image_folder}/*.png"))

# 가로선의 y좌표를 찾는 함수
def find_horizontal_line_y(image_path):
    # 이미지 읽기
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 초록색 마스크 생성
    mask = cv2.inRange(hsv, green_lower, green_upper)

    # 컨투어(윤곽선) 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 가장 큰 컨투어 찾기 (초록색 선일 가능성이 높음)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)  # 바운딩 박스 정보
        return y + h // 2  # 가로선의 중앙 y좌표
    return None  # 선이 없으면 None 반환

# 이미지 정렬
image_y_positions = []
for image_path in images:
    y = find_horizontal_line_y(image_path)
    if y is not None:
        image_y_positions.append((image_path, y))

# y좌표를 기준으로 내림차순 정렬
sorted_images = sorted(image_y_positions, key=lambda x: x[1], reverse=True)

# 결과 저장
os.makedirs(output_folder, exist_ok=True)
for idx, (image_path, y) in enumerate(sorted_images):
    image = cv2.imread(image_path)
    new_filename = os.path.join(output_folder, f"{idx+1:03d}.png")
    cv2.imwrite(new_filename, image)

print(f"Images sorted by horizontal line position and saved in {output_folder}")
