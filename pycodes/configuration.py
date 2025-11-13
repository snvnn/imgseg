import os
import torch

# 고정되어야 하는 것. 바꾸지 말 것
IMAGE_SIZE = (256, 256)
N_CLASSES = 3  # trimap은 각 픽셀이 배경, 가장자리, 동물부분 이렇게 3개 중 하나의 클래스로 배정된 것

# Seeds
torch.manual_seed(0)

# Device settings
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data paths
IMAGE_PATH = 'train/images'
MAP_PATH = 'train/trimaps'

# Learning parameters
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
EPOCHS = 1000

# Output paths
OUTPUT_PATH = 'train/output'
if not os.path.exists(OUTPUT_PATH):
  os.mkdir(OUTPUT_PATH)
MODEL_PATH = os.path.join(OUTPUT_PATH, 'model_weight.pth')
HISTORY_PATH = os.path.join(OUTPUT_PATH, 'history.pickle')
HISTORY_PLOT_PATH = os.path.join(OUTPUT_PATH, 'history.png')  # 드라이브에 저장하고 싶다면 경로를 변경해야 함
HISTORY_LOGPLOT_PATH = os.path.join(OUTPUT_PATH, 'loghistory.png') 
PRED_PLOT_PATH = os.path.join(OUTPUT_PATH, 'pred.png')