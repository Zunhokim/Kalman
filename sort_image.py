import cv2
import glob
import numpy as np
import os

# Parameters
image_folder = '/Users/junho_kim/Downloads/crossline_image'  # 이미지가 있는 폴더 경로
output_folder = '/Users/junho_kim/Downloads/crossline_image_sorted'  # 결과 파일 저장 폴더
corner_size = 50  # 모퉁이 크기 (픽셀)

# 이미지 파일 가져오기
images = sorted(glob.glob(f"{image_folder}/*.png"))

# 모퉁이 평균값 계산 함수
def calculate_corner_features(image):
    h, w, _ = image.shape

    # 네 모퉁이 추출
    corners = {
        "top_left": image[:corner_size, :corner_size],  # 좌측 상단
        "top_right": image[:corner_size, w-corner_size:],  # 우측 상단
        "bottom_left": image[h-corner_size:, :corner_size],  # 좌측 하단
        "bottom_right": image[h-corner_size:, w-corner_size:]  # 우측 하단
    }

    # 각 모퉁이의 평균값 계산 (RGB 채널의 평균값)
    features = {key: np.mean(corner) for key, corner in corners.items()}
    return features

# 유사도 비교 함수 (특징 벡터 간 차이 계산)
def calculate_similarity(feature1, feature2):
    # 유사도: 각 모퉁이의 평균값 차이의 절댓값을 합산
    return sum(abs(feature1[key] - feature2[key]) for key in feature1)

# 특징 추출
image_features = {}
for image_path in images:
    image = cv2.imread(image_path)
    features = calculate_corner_features(image)
    image_features[image_path] = features

# 유사도를 기준으로 이미지 순서 정렬
sorted_images = sorted(
    images,
    key=lambda x: (
        # 첫 번째 이미지를 기준으로 정렬
        sum(
            calculate_similarity(image_features[x], image_features[y])
            for y in images
        )
    )
)

# 파일명 재배치 및 저장
os.makedirs(output_folder, exist_ok=True)
for idx, image_path in enumerate(sorted_images):
    image = cv2.imread(image_path)
    new_filename = os.path.join(output_folder, f"sorted_{idx+1:03d}.png")
    cv2.imwrite(new_filename, image)

print(f"Images reordered and saved in {output_folder}")
