import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from configuration import IMAGE_PATH, IMAGE_SIZE, BATCH_SIZE, MAP_PATH, N_CLASSES, DEVICE, MODEL_PATH
from dataset import TrimapsDataset
from model import ImprovedUNet


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

# Data paths
TEST_PATH = 'images'
SUBMISSION_FILE_PATH = 'submission/submission.csv'

test_set = TrimapsDataset(TEST_PATH, '', test=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

model = ImprovedUNet(in_channel=3, out_channel=N_CLASSES, img_size=IMAGE_SIZE,
                     base_ch=32, norm="bn", se=True, drop=0.1, use_aspp=True)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE)

get_rle(model, test_loader, SUBMISSION_FILE_PATH)