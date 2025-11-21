from torch.utils.data import DataLoader
from dataset import TrimapsDataset
from configuration import IMAGE_PATH, BATCH_SIZE, MAP_PATH, MODEL_PATH, DEVICE, CHECKPOINT, LEARNING_RATE, WEIGHT_DECAY, LR_SCHEDULING_FACTOR, LR_SCHEDULING_PATIENCE, MIN_SCHEDULING_LR, MODEL
from train import train

import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

train_set = TrimapsDataset(IMAGE_PATH, MAP_PATH, test=False)
test_set = TrimapsDataset(IMAGE_PATH, MAP_PATH, test=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

model = MODEL
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=LR_SCHEDULING_FACTOR,
    patience=LR_SCHEDULING_PATIENCE,
    min_lr=MIN_SCHEDULING_LR
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
train(model, train_loader, test_loader, optimizer=optimizer, scheduler=scheduler, start_epoch=start_epoch)
