from torch.utils.data import DataLoader
from dataset import TrimapsDataset
from configuration import IMAGE_PATH, BATCH_SIZE, MAP_PATH, MODEL_PATH, DEVICE, CHECKPOINT, LEARNING_RATE, LR_SCHEDULING_FACTOR, LR_SCHEDULING_PATIENCE, MIN_SCHEDULING_LR, VERBOSE, MODEL
from model import ImprovedUNet
from train import train

import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

train_set = TrimapsDataset(IMAGE_PATH, MAP_PATH, test=False)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

model = MODEL
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=LR_SCHEDULING_FACTOR,
    patience=LR_SCHEDULING_PATIENCE,
    min_lr=MIN_SCHEDULING_LR,
    verbose=VERBOSE,
)

start_epoch = 0

if os.path.exists(MODEL_PATH) and CHECKPOINT:
    print("Loading checkpoint...")
    checkpoint = torch.load(MODEL_PATH + ".ckpt", map_location=DEVICE)
    
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    start_epoch = checkpoint["epoch"] + 1
    
    print(f"Resumed from epoch {start_epoch}")

model = model.to(DEVICE)

# train 함수 실행
train(model, train_loader, optimizer=optimizer, scheduler=scheduler, start_epoch=start_epoch)