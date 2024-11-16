import cv2
import glob
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Parameters
image_folder = '/Users/junho_kim/Downloads/Prediction_path_image'  # 이미지가 있는 폴더 경로
output_folder = '/Users/junho_kim/Downloads/sorted_image'  # 결과 파일 저장 폴더
left_width = 350  # 좌측 영역 너비 (픽셀 단위)
right_width = 200  # 우측 영역 너비 (픽셀 단위)
corner_size = 50  # 네 모퉁이 비교 영역 크기 (픽셀 단위)
weights = {
    "brightness": 0.15,  # 밝기 가중치
    "contrast": 0.15,    # 대비 가중치
    "color": 0.4,       # 색상 정보 가중치
    "edge": 0.2,        # 에지 가중치
}  # 각 지표의 중요도를 설정 (합이 1이어야 함)

# 이미지 파일 가져오기
images = sorted(glob.glob(f"{image_folder}/*.png"))

# 밝기 계산 함수
def calculate_brightness(image):
    return np.mean(image)

# 대비 계산 함수 (픽셀 값 표준 편차)
def calculate_contrast(image):
    return np.std(image)

# 색상 정보 계산 함수 (HSV 사용)
def calculate_colorfulness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_mean = np.mean(hsv[:, :, 0])  # Hue 평균
    saturation_mean = np.mean(hsv[:, :, 1])  # Saturation 평균
    return hue_mean, saturation_mean

# 에지 계산 함수 (Canny 에지 검출 후 픽셀 비율 계산)
def calculate_edge_strength(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges) / edges.size  # 에지가 있는 픽셀 비율

# 구조적 유사성 비교 함수
def calculate_ssim(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

# 이미지 정렬을 위한 지표 계산
image_features = []
for image_path in images:
    image = cv2.imread(image_path)
    
    # 밝기, 대비, 색상, 에지 계산
    brightness = calculate_brightness(image)
    contrast = calculate_contrast(image)
    hue_mean, saturation_mean = calculate_colorfulness(image)
    edge_strength = calculate_edge_strength(image)
    
    # 지표 저장
    image_features.append({
        "path": image_path,
        "brightness": brightness,
        "contrast": contrast,
        "hue": hue_mean,
        "saturation": saturation_mean,
        "edge": edge_strength
    })

# 정렬 기준 설정 (여기선 brightness + contrast + colorfulness + edge)
def calculate_combined_score(features):
    brightness_score = features["brightness"] * weights["brightness"]
    contrast_score = features["contrast"] * weights["contrast"]
    color_score = (features["hue"] + features["saturation"]) * weights["color"]
    edge_score = features["edge"] * weights["edge"]
    return brightness_score + contrast_score + color_score + edge_score

# 정렬
sorted_features = sorted(image_features, key=calculate_combined_score, reverse=True)

# 결과 저장
os.makedirs(output_folder, exist_ok=True)
for idx, features in enumerate(sorted_features):
    image = cv2.imread(features["path"])
    new_filename = os.path.join(output_folder, f"{idx+1:03d}.png")
    cv2.imwrite(new_filename, image)

    # 디버그: 정렬된 결과 출력
    print(f"Image: {os.path.basename(features['path'])}, Combined Score: {calculate_combined_score(features):.2f}")

print(f"Images sorted using multiple metrics and saved in {output_folder}")
