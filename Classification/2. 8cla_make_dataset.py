'''
1. Total 라벨링 + 리사이즈
2. 불균형 해소
3. 증강 + 최종 라벨링 (1번 코드 다시 사용)
4. train, val 분리
'''


# 라벨링
import os

def map_age_to_category(age):
    """ Map a given age to the corresponding category. """
    if 0 <= age <= 20:
        return 'u20'
    elif 21 <= age <= 30:
        return '20s'
    elif 31 <= age <= 40:
        return '30s'
    elif 41 <= age <= 50:
        return '40s'
    else:
        return 'over50'
    
# 데이터 가공
def create_label_file(dataset_dir, output_file):
    """
    Create labels.txt for Caffe training from All-Age-Faces dataset, sorted by age.
    :param dataset_dir: Directory containing original images.
    :param output_file: Path to save the labels.txt file.
    """
    labels = []

    # 이미지 파일의 경로와 나이를 리스트에 저장
    for root, _, files in os.walk(dataset_dir):
        for img_file in files:
            if 'A' in img_file:
                # 파일명에서 나이 추출
                try:
                    age = int(img_file.split('A')[1][:2])  # 'A25'에서 25 추출
                    img_path = os.path.join(root, img_file)
                    cat = map_age_to_category(age)
                    labels.append((img_path, cat))
                except (ValueError, IndexError):
                    # 나이 추출 실패시 스킵
                    continue
            elif '_' in img_file:
                # 파일명에서 나이 추출
                try:
                    age = int(os.path.basename(img_file).split('_')[0])  # 파일명에서 나이 추출
                    img_path = os.path.join(root, img_file)
                    cat = map_age_to_category(age)
                    labels.append((img_path, cat))
                except (ValueError, IndexError):
                    # 나이 추출 실패시 스킵
                    continue

    # 나이 기준으로 정렬
    labels.sort(key=lambda x: x[1])

    # 정렬된 데이터를 파일에 저장
    with open(output_file, 'w') as file:
        for img_path, age in labels:
            file.write(f"{img_path}\t{age}\n")

# 사용 예시
dataset_dir = "/raid/co_show02/JY/CNN/Total"
rawoutput_file = "/raid/co_show02/JY/CNN/txt/ca_rawlabels.txt"
create_label_file(dataset_dir, rawoutput_file)



# import os
# import random
# from collections import defaultdict

# def map_age_to_category(age):
#     """ Map a given age to the corresponding category. """
#     if 0 <= age <= 20:
#         return 'u20'
#     elif 21 <= age <= 30:
#         return '20s'
#     elif 31 <= age <= 40:
#         return '30s'
#     elif 41 <= age <= 50:
#         return '40s'
#     else:
#         return 'over50'

# def balance_classes(labels_file, output_labels_file):
#     """
#     Balance the dataset by adjusting the number of samples for each class to be equal to the average count.
#     :param labels_file: Path to labels.txt file.
#     :param output_labels_file: Path to save balanced labels.txt file.
#     """
#     # 클래스별로 데이터 분류
#     class_data = defaultdict(list)
#     with open(labels_file, 'r') as file:
#         for line in file:
#             img_path, age_str = line.strip().rsplit('\t', 1)
#             age = int(age_str)
#             label = map_age_to_category(age)  # 나이를 카테고리로 변환
#             class_data[label].append(img_path)
    
#     # 각 클래스의 평균 샘플 수 계산
#     total_samples = sum(len(img_paths) for img_paths in class_data.values())
#     avg_count = total_samples // len(class_data)
    
#     balanced_data = []
    
#     # 클래스별 데이터 균형 맞추기
#     for label, img_paths in class_data.items():
#         current_count = len(img_paths)
#         if current_count > avg_count:
#             # 언더샘플링: 평균 수보다 많은 경우 평균 수만큼 샘플링
#             selected_paths = random.sample(img_paths, avg_count)
#         elif current_count < avg_count:
#             # 오버샘플링: 평균 수보다 적은 경우 중복하여 평균 수까지 만듦
#             additional_paths = random.choices(img_paths, k=avg_count - current_count)
#             selected_paths = img_paths + additional_paths
#         else:
#             selected_paths = img_paths
        
#         random.shuffle(selected_paths)  # 섞기
#         for img_path in selected_paths:
#             balanced_data.append(f"{img_path}\t{label}\n")
    
#     # Balanced 데이터셋 라벨 파일 저장
#     with open(output_labels_file, 'w') as file:
#         file.writelines(balanced_data)
    
#     print(f"Balanced labels saved to {output_labels_file}")

# # 실행
# balance_classes("/raid/co_show02/JY/CNN/txt/rawlabels.txt", "/raid/co_show02/JY/CNN/txt/balanced_labels.txt")


# ###############
# #######

# # 어그멘테이션
# import os
# import random
# import albumentations as A
# import cv2
# from collections import defaultdict

# # 증강 파이프라인 정의
# augmentations = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.2),
#     A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
#     A.GaussianBlur(blur_limit=(3, 7), p=0.3),
#     A.Affine(shear=15, p=0.5),  # 평행사변형처럼 보이도록 하는 변형 (shear 값 조정 가능)
#     A.Perspective(scale=(0.05, 0.1), p=0.3)  # 약간의 원근 변환을 추가하여 왜곡 효과
# ])

# def augment_balanced_classes(labels_file, output_dir, augmentations, augment_per_class):
#     """
#     Augment images from each class to maintain balanced classes.
#     :param labels_file: Path to labels.txt file containing image paths and labels.
#     :param output_dir: Directory to save augmented images.
#     :param augmentations: Albumentations augmentation pipeline.
#     :param augment_per_class: Number of augmented images to generate per class.
#     """
    
#     # 클래스별로 데이터 분류
#     class_data = defaultdict(list)
#     with open(labels_file, 'r') as file:
#         for line in file:
#             img_path, label = line.strip().rsplit('\t', 1)
#             class_data[label].append(img_path)
    
#     augmented_data = []
#     for label, img_paths in class_data.items():
#         # 각 클래스에 대해 지정된 수만큼 증강 수행
#         for i in range(augment_per_class):
#             img_path = random.choice(img_paths)  # 랜덤으로 이미지를 선택하여 증강
#             image = cv2.imread(img_path)
            
#             if image is None:
#                 print(f"Image not found or could not be read: {img_path}")
#                 continue
            
#             augmented = augmentations(image=image)
#             aug_image = augmented['image']
#             aug_img_name = f"2aug{os.path.splitext(os.path.basename(img_path))[0]}{i}.jpg"
#             aug_img_path = os.path.join(output_dir, aug_img_name)
#             cv2.imwrite(aug_img_path, aug_image)
#             augmented_data.append(f"{aug_img_path}\t{label}\n")

#     # 원본 데이터와 증강된 데이터를 합쳐서 최종 라벨 파일 생성
#     final_labels_file = os.path.join(output_dir, "final_labels.txt")
#     with open(labels_file, 'r') as file:
#         original_data = file.readlines()
    
#     with open(final_labels_file, 'w') as file:
#         file.writelines(original_data)  # 원본 데이터 추가
#         file.writelines(augmented_data)  # 증강된 데이터 추가

#     print(f"Augmented images saved to {output_dir}")
#     print(f"Final labels saved to {final_labels_file}")

# augment_balanced_classes("/raid/co_show02/JY/CNN/txt/balanced_labels.txt", "/raid/co_show02/JY/CNN/up_aug", augmentations, augment_per_class=1000)
# # /raid/co_show02/JY/v2_caffe/augmentation/final_labels.txt
