import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import glob

# Parameters
image_folder = '/Users/junho_kim/Downloads/Prediction_path_image'  # 이미지가 있는 폴더 경로
output_folder = '/Users/junho_kim/Downloads/sorted_image'  # 결과 파일 저장 폴더
images = sorted(glob.glob(f"{image_folder}/*.png"))  # 이미지 파일 읽기
left_width = 250  # 좌측 비교 영역의 너비
right_width = 250  # 우측 비교 영역의 너비

# 좌우 영역 특징 추출 함수
def extract_features(image_path, left_width, right_width):
    # 이미지 로드
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape

    # 좌측 영역 추출
    left_area = img[:, :left_width]
    # 우측 영역 추출
    right_area = img[:, -right_width:]

    # 1. 밝기 평균
    left_brightness = np.mean(left_area)
    right_brightness = np.mean(right_area)

    # 2. 컨트라스트 계산 (픽셀 간 차이의 표준 편차)
    left_contrast = np.std(left_area)
    right_contrast = np.std(right_area)

    # 3. 엣지 밀도 (Canny 엣지 검출 사용)
    left_edges = cv2.Canny(left_area, 50, 150)
    right_edges = cv2.Canny(right_area, 50, 150)
    left_edge_density = np.sum(left_edges) / (left_area.shape[0] * left_area.shape[1])
    right_edge_density = np.sum(right_edges) / (right_area.shape[0] * right_area.shape[1])

    # 4. 구조적 유사도 (SSIM)
    ssim_score = ssim(left_area, right_area)

    # 5. 히스토그램 비교 (상관 계수 방식)
    left_hist = cv2.calcHist([left_area], [0], None, [256], [0, 256])
    right_hist = cv2.calcHist([right_area], [0], None, [256], [0, 256])
    left_hist = cv2.normalize(left_hist, left_hist).flatten()
    right_hist = cv2.normalize(right_hist, right_hist).flatten()
    hist_correlation = cv2.compareHist(left_hist, right_hist, cv2.HISTCMP_CORREL)

    return {
        "left_brightness": left_brightness,
        "right_brightness": right_brightness,
        "left_contrast": left_contrast,
        "right_contrast": right_contrast,
        "left_edge_density": left_edge_density,
        "right_edge_density": right_edge_density,
        "ssim_score": ssim_score,
        "hist_correlation": hist_correlation,
    }

# 이미지 특징 계산
image_features = []
for image_path in images:
    features = extract_features(image_path, left_width, right_width)
    image_features.append((image_path, features))

# 정렬 기준: 각 지표를 가중합하여 최종 순위 계산
def calculate_weighted_score(features, weights):
    score = (
        weights["brightness"] * abs(features["left_brightness"] - features["right_brightness"])
        + weights["contrast"] * abs(features["left_contrast"] - features["right_contrast"])
        + weights["edge_density"] * abs(features["left_edge_density"] - features["right_edge_density"])
        - weights["ssim"] * features["ssim_score"]  # SSIM은 유사할수록 낮은 점수로
        - weights["hist_correlation"] * features["hist_correlation"]  # 상관 계수도 유사할수록 낮은 점수로
    )
    return score

# 가중치 설정 (사용자가 조정 가능)
weights = {
    "brightness": 1.0,
    "contrast": 1.0,
    "edge_density": 1.5,
    "ssim": 2.0,
    "hist_correlation": 2.0,
}

# 정렬
sorted_images = sorted(
    image_features,
    key=lambda x: calculate_weighted_score(x[1], weights)
)

# 정렬된 이미지 저장
for idx, (image_path, features) in enumerate(sorted_images):
    img = cv2.imread(image_path)
    filename = f"sorted_{idx:03d}.png"
    cv2.imwrite(f"{output_folder}/{filename}", img)

print(f"Images sorted and saved to {output_folder}")
