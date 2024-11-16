import cv2
import glob
import numpy as np
import os

# Parameters
image_folder = '/Users/junho_kim/Downloads/crossline_image'  # 이미지가 있는 폴더 경로
output_folder = '/Users/junho_kim/Downloads/crossline_image_sorted'  # 결과 파일 저장 폴더
corner_size = 74  # 모퉁이 크기 (픽셀)
similarity_threshold = 30  # 유사도 임계값 (작을수록 더 엄격하게 분류)
group_counter = 1  # 그룹 번호 초기값

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

# 그룹화 함수
def assign_to_group(image_path, features, groups):
    global group_counter

    # 현재 이미지의 특징 벡터와 기존 그룹 비교
    for group_id, group_images in groups.items():
        # 그룹의 대표 이미지와 유사도 비교
        representative_image = group_images[0]  # 그룹 내 첫 번째 이미지를 대표로 사용
        similarity = calculate_similarity(features, representative_image['features'])

        if similarity < similarity_threshold:
            # 유사하면 해당 그룹에 추가
            groups[group_id].append({"path": image_path, "features": features})
            return

    # 유사한 그룹이 없으면 새로운 그룹 생성
    groups[group_counter] = [{"path": image_path, "features": features}]
    group_counter += 1

# 이미지 그룹화
groups = {}
for image_path in images:
    image = cv2.imread(image_path)
    features = calculate_corner_features(image)
    assign_to_group(image_path, features, groups)

# 결과 저장
os.makedirs(output_folder, exist_ok=True)
for group_id, group_images in groups.items():
    group_folder = os.path.join(output_folder, f"group_{group_id}")
    os.makedirs(group_folder, exist_ok=True)

    # 그룹 내 이미지 저장
    for idx, image_info in enumerate(group_images):
        image = cv2.imread(image_info['path'])
        new_filename = os.path.join(group_folder, f"{idx+1:03d}.png")
        cv2.imwrite(new_filename, image)

print(f"Images grouped and saved in {output_folder}")
