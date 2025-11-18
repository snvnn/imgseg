from model import ImprovedUNet
from configuration import N_CLASSES, IMAGE_SIZE, MODEL_PATH, DEVICE, IMAGE_PATH, MAP_PATH, PRED_PLOT_PATH, MODEL

import os
import torch
import random
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from PIL import Image

# 저장된 모델 weight 로드
model = MODEL
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE)
model.eval()

# 이미지 경로의 리스트 생성
img_path_list = sorted([
    os.path.join(IMAGE_PATH, name)
    for name in os.listdir(IMAGE_PATH)
    if name.endswith('.jpg')
])

# 맵 경로의 리스트 생성
map_path_list = sorted([
    os.path.join(MAP_PATH, name)
    for name in os.listdir(MAP_PATH)
    if name.endswith('.png')
])

# 랜덤한 세개의 인덱스 추출 후 이미지와 맵의 경로 저장
indices = random.sample(range(len(img_path_list) - 1), 3)
image_map_paths = {img_path_list[idx]: map_path_list[idx] for idx in indices}

# 이미지 transform 정의. 데이터셋 클래스와 같아야 함
transform = transforms.Compose([
transforms.Resize(IMAGE_SIZE),
transforms.ToTensor()
])

with torch.no_grad():
  figure, ax = plt.subplots(3, 3, figsize=(12, 12), subplot_kw={'xticks': [], 'yticks': []})

  for i, (image, map) in enumerate(image_map_paths.items()):
    # 이미지 로드 후 transform. 데이터셋 클래스와 같은 작업
    image = Image.open(image).convert('RGB')
    image = transform(image).to(DEVICE)
    image = torch.unsqueeze(image, dim=0) # 배치 dimension 추가

    # 모델 통과 후 예측 맵 생성
    pred = model(image).squeeze()
    pred = torch.argmax(pred, dim=0) # logit이 가장 큰 class 선택
    pred = pred.cpu().numpy()
    pred = pred * 255   # 이미지로 표시하기 위해 255 곱함

    # 정답 맵 로드
    map = Image.open(map).convert('L')
    map = transform(map)
    map = torch.squeeze(map, dim=0)
    map = map * 255
    map -= 1

    # 이미지 plot
    image = image.squeeze().permute(1, 2, 0).cpu()
    ax[i, 0].imshow(image)
    ax[i, 1].imshow(map)
    ax[i, 2].imshow(pred)

  # Save the plot
  figure.savefig(PRED_PLOT_PATH)