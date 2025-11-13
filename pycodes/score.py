from torch.utils.data import DataLoader
from configuration import IMAGE_PATH, IMAGE_SIZE, BATCH_SIZE, MAP_PATH, N_CLASSES, DEVICE, MODEL_PATH
from dataset import TrimapsDataset
from model import ImprovedUNet
from train import train

train_set = TrimapsDataset(IMAGE_PATH, MAP_PATH, test=False)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

model = ImprovedUNet(in_channel=3, out_channel=N_CLASSES, img_size=IMAGE_SIZE,
                     base_ch=32, norm="bn", se=True, drop=0.1, use_aspp=True)
model = model.to(DEVICE)

# train 함수 실행
train(model, train_loader)