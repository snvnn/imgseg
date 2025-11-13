from model import ImprovedUNet
from configuration import N_CLASSES, IMAGE_SIZE, MODEL_PATH, DEVICE
from run import train_loader
import torch

model = ImprovedUNet(3, N_CLASSES, IMAGE_SIZE)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE)

eval(model, train_loader)