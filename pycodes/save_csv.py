import os
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from configuration import BATCH_SIZE, DEVICE, MODEL_PATH, MODEL, SUBMISSION_DIR
from dataset import TrimapsDataset


# 테스트 데이터 다운로드 & unzip
#drive.mount('/content/drive')
#!cp "/content/drive/MyDrive/Colab Notebooks/project/dataset/test.zip" /content/
#!unzip /content/test.zip -d /content/testset

# RLE (Run Length Encoding) 함수. 파일 제출 형식을 맞추기 위해 필요. 수정하지 말 것
def rle_encode(mask):
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def get_rle(model, test_loader, save_path):
  # Ensure the output directory exists when called from training threads
  save_dir = os.path.dirname(save_path)
  if save_dir:
    os.makedirs(save_dir, exist_ok=True)

  model.eval()

  image_ids = []
  image_paths = []
  class_ids = []
  rles = []

  with torch.no_grad():
    for i, (images, data_id) in enumerate(test_loader):
      images = images.to(DEVICE)

      outputs = model(images)
      preds = torch.argmax(outputs, dim=1)
      preds = preds.cpu().numpy()  # numpy로 변환

      for b in range(preds.shape[0]):
          pred_mask = preds[b] + 1  # 모델에서 편의를 위해 0, 1, 2로 변환했었기 때문에 되돌리기 위해 +1
          image_id = 'TEST_'+data_id[b]

          # 클래스별 RLE 인코딩 (1=foreground, 2=border, 3=background)
          for class_id in [1, 2, 3]:
              mask = (pred_mask == class_id).astype(np.uint8)
              rle = rle_encode(mask) if mask.sum() > 0 else ''

              image_ids.append(image_id)
              class_ids.append(class_id)
              rles.append(rle)

    # DataFrame 저장
    df = pd.DataFrame({
        'image_id': image_ids,
        'class_id': class_ids,
        'rle': rles
    })
    df.to_csv(save_path, index=False)
    print(f"RLE CSV file saved : {save_path}")

    return df

# Standalone generation helper
TEST_PATH = 'images'
CHECKPOINT_DIR = os.path.dirname(MODEL_PATH)

def _checkpoint_to_csv_name(checkpoint_path):
  base = os.path.basename(checkpoint_path)
  # Strip all suffixes (e.g., .pth.ckpt -> model_weight)
  while True:
    stem, ext = os.path.splitext(base)
    if not ext:
      break
    base = stem
  return os.path.join(SUBMISSION_DIR, f"{base}.csv")

def _list_checkpoints(checkpoint_dir):
  if not os.path.isdir(checkpoint_dir):
    raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

  allowed_ext = ('.pth', '.pt', '.ckpt')
  checkpoints = [
    os.path.join(checkpoint_dir, name)
    for name in os.listdir(checkpoint_dir)
    if name.lower().endswith(allowed_ext)
  ]
  return sorted(checkpoints)

def load_trained_model(checkpoint_path):
  if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(
      f"Trained weights not found at {checkpoint_path}. "
      "Run training first to generate this file or place checkpoints in the directory."
    )

  model = deepcopy(MODEL)
  state_dict = torch.load(checkpoint_path, map_location=DEVICE)
  if isinstance(state_dict, dict) and "model_state" in state_dict:
    state_dict = state_dict["model_state"]
  model.load_state_dict(state_dict)
  return model

def build_test_loader():
  test_set = TrimapsDataset(TEST_PATH, '', test=True)
  return DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

def main():
  test_loader = build_test_loader()

  checkpoints = _list_checkpoints(CHECKPOINT_DIR)
  if not checkpoints:
    raise FileNotFoundError(f"No checkpoint files found in {CHECKPOINT_DIR}")

  os.makedirs(SUBMISSION_DIR, exist_ok=True)

  for checkpoint_path in checkpoints:
    model = load_trained_model(checkpoint_path).to(DEVICE)
    submission_path = _checkpoint_to_csv_name(checkpoint_path)
    get_rle(model, test_loader, submission_path)

if __name__ == "__main__":
  main()
