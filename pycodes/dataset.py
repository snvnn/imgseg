import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from configuration import IMAGE_SIZE
import torchvision.transforms as transforms
import os

class TrimapsDataset(Dataset):
  def __init__(self, img_path, map_path='', test=False):
    super().__init__()

    self.istest=test

    # 이미지 경로의 리스트 생성
    self.img_path_list = sorted([
        os.path.join(img_path, name)
        for name in os.listdir(img_path)
        if name.endswith('.jpg')
    ])

    # 맵 경로의 리스트 생성, train 데이터에만 gt 맵 존재
    if not self.istest:
      self.map_path_list = sorted([
          os.path.join(map_path, name)
          for name in os.listdir(map_path)
          if name.endswith('.png')
      ])

    # 이미지 transform 정의 (리사이즈, 텐서로 변환)
    self.transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
    ])

  def __len__(self):
    return len(self.img_path_list)

  def __getitem__(self, idx):

    image_path = self.img_path_list[idx]
    # 데이터 id 추출. 결과제출시 필요
    data_id = image_path.split('/')[-1].split('.')[0].split('_')[-1]

    # 이미지 열어 RGB로 변경
    image = Image.open(image_path).convert('RGB')
    # 이미지 transform 적용
    image = self.transform(image)

    # train 데이터에만 gt 맵 존재
    if not self.istest:
      map_path = self.map_path_list[idx]
      # 맵 열어 grayscale 로 변경
      map = Image.open(map_path).convert('L')
      # 맵 transform 적용
      map = self.transform(map)
      # map은 0과 1 사이의 값으로 되어있으므로 255를 곱해 정수로 만듦
      map = map * 255
      map = map.squeeze().to(torch.int64)
      # GT 라벨은 1, 2, 3인데 분류 편의성을 위해 0, 1, 2로 변경
      map -= 1
      return image, map

    return image, data_id  # 채점을 위해 테스트 모드에서는 데이터 ID와 함께 반환

