from model import ImprovedUNet, DeepUNet #, OptimizedUNet
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
IMAGE_PATH = 'open/train/images'
MAP_PATH = 'open/train/trimaps'

# Learning parameters
# Bump batch size to better utilize ~8GB GPU
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 1000
MIN_DELTA = 5e-4  # 개선으로 인정할 최소 손실 감소량
BEST_LOSS = float('inf')
PATIENCE = 6     # PATIENCE > 0 일 때 Early Stop 동작

MODEL = DeepUNet(
    in_channel=3,
    out_channel=N_CLASSES,
    img_size=IMAGE_SIZE,
    base_ch=32,
    norm="bn",
    se=True,
    drop=0.1,
    use_aspp=True,
    # use_attention=True
)  # 사용 모델 선택: ImprovedUNet, DeepUNet, OptimizedUNet 중 선택

CHECKPOINT = False          # True: 기존에 생성된 모델 가중치를 불러와 학습, False: 처음부터 학습
LR_SCHEDULING_FACTOR = 0.3  # 학습률 스케쥴러에 적용할 학습률 감소 정도
LR_SCHEDULING_PATIENCE = 2  # 학습률 스케쥴러에 적용할 patience
MIN_SCHEDULING_LR = 1e-6    # LR이 줄어들 수 있는 최소값

LAMBDA = 0.7                # Dice Loss Fuction 반영 비율. 범위: 0.3 ~ 1.0

# Output paths
OUTPUT_PATH = 'output/'
if not os.path.exists(OUTPUT_PATH):
  os.mkdir(OUTPUT_PATH)
MODEL_PATH = os.path.join(OUTPUT_PATH, 'model_weight.pth')
HISTORY_PATH = os.path.join(OUTPUT_PATH, 'history.pickle')
HISTORY_PLOT_PATH = os.path.join(OUTPUT_PATH, 'history.png')  # 드라이브에 저장하고 싶다면 경로를 변경해야 함
HISTORY_LOGPLOT_PATH = os.path.join(OUTPUT_PATH, 'loghistory.png') 
PRED_PLOT_PATH = os.path.join(OUTPUT_PATH, 'pred.png')
SUBMISSION_DIR = 'submission/'
