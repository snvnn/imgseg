from dataset import TrimapsDataset
from configuration import BATCH_SIZE, N_CLASSES, IMAGE_SIZE, MODEL_PATH, DEVICE, HISTORY_PATH, HISTORY_LOGPLOT_PATH, HISTORY_PLOT_PATH, MODEL
from model import ImprovedUNet
from save_csv import get_rle

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle



# Data paths
TEST_PATH = 'images'
SUBMISSION_FILE_PATH = 'submission/submission.csv'

test_set = TrimapsDataset(TEST_PATH, '', test=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

model = MODEL
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE)

get_rle(model, test_loader, SUBMISSION_FILE_PATH)

TEST_PATH = 'images'
SUBMISSION_FILE_PATH = 'submission/submission.csv'

test_set = TrimapsDataset(TEST_PATH, '', test=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

model = ImprovedUNet(in_channel=3, out_channel=N_CLASSES, img_size=IMAGE_SIZE,
                     base_ch=32, norm="bn", se=True, drop=0.1, use_aspp=True)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE)

get_rle(model, test_loader, SUBMISSION_FILE_PATH)

# 저장한 Loss 히스토리 로드
with open(HISTORY_PATH, 'rb') as f:
    model_history = pickle.load(f)

# 그래프 만들어 이미지로 저장
plt.figure()
plt.plot(model_history['train_loss'], label='Train loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig(HISTORY_PLOT_PATH)

# 저장한 Loss 히스토리 로드
with open(HISTORY_PATH, 'rb') as f:
    model_history = pickle.load(f)

# 그래프 만들어 이미지로 저장
plt.figure()
plt.plot(model_history['train_loss'], label='Train loss')

# 로그 스케일 설정
plt.yscale('log')

plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.legend(loc='upper right')
plt.title('Training Loss (Log Scale)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.savefig(HISTORY_LOGPLOT_PATH, dpi=200, bbox_inches='tight')
plt.close()