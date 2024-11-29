import os
import PIL.Image as Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, label_file, train=False):
        super(CustomDataset, self).__init__()
        self.label_file = label_file
        self.age_range = (1, 6)  # 나이 범주 1~5 설정
        self.transform = None
        self.is_train = train
        self.data = []
        
        # 파일에서 경로와 나이 정보 읽어오기
        with open(self.label_file, 'r') as file:
            for line in file:
                path, age_category = line.strip().split('\t')
                self.data.append((path, age_category))
        
        self.num_imgs = len(self.data)
        self.idx = np.array([i for i in range(self.__len__())], dtype=int)
        self.Image_Transform()

    def Image_Transform(self):
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((60, 60)),
                transforms.Normalize(0, 1),
                transforms.RandomCrop(60),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((60, 60)),
            ])
        pass

    def map_age_to_category(self, age_category):
        # 문자열을 나이 범주에 맞는 숫자 값으로 매핑
        if age_category == 'u20':
            return 1
        elif age_category == '20s':
            return 2
        elif age_category == '30s':
            return 3
        elif age_category == '40s':
            return 4
        else:  # 'over50'
            return 5

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        img_path, age_category = self.data[idx]
        img = Image.open(img_path)
        img = self.transform(img)
        age_label = int(self.map_age_to_category(age_category))
        label = torch.zeros(5, 2)
        # 나이별로 이진 레이블을 생성 (순차적으로 누적하여 채움)
        label[:age_label - 1] = torch.tensor([1, 0])  # 나이보다 작은 부분은 [1, 0]으로 설정
        label[age_label - 1:] = torch.tensor([0, 1])  # 나이보다 크거나 같은 부분은 [0, 1]으로 설정
        return img, label, age_label