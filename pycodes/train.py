import torch
from copy import deepcopy
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import time
import pickle
from configuration import LEARNING_RATE, EPOCHS,HISTORY_PATH, DEVICE, MODEL_PATH


def train(model, train_loader):

  model.train()

  loss_fn = CrossEntropyLoss()
  optim = Adam(model.parameters(), lr=LEARNING_RATE)

  history = {'train_loss': []}
  start_time = time.time()

  # Early stopping 설정
  patience = 10          # 개선이 없을 때 허용할 최대 epoch 수
  min_delta = 1e-4       # 개선으로 인정할 최소 손실 감소량
  best_loss = float('inf')
  best_state_dict = None
  epochs_no_improve = 0

  for epoch in range(EPOCHS):
    print('EPOCH: {}'.format(epoch))
    train_loss = 0.

    for i, (images, targets) in enumerate(train_loader):
      images, targets = images.to(DEVICE), targets.to(DEVICE)

      pred = model(images)
      loss = loss_fn(pred, targets)

      optim.zero_grad()
      loss.backward()
      optim.step()

      train_loss += loss*len(images)

    avg_train_loss = train_loss/len(train_loader.dataset)
    print('TRAIN LOSS: {}'.format(avg_train_loss))
    history['train_loss'].append(avg_train_loss.cpu().detach().numpy())

    # Early stopping 로직 (train loss 기준)
    current_loss = avg_train_loss.item()
    if best_loss - current_loss > min_delta:
      best_loss = current_loss
      best_state_dict = deepcopy(model.state_dict())
      epochs_no_improve = 0
      print(f'  -> improvement, best_loss updated to {best_loss:.6f}')
    else:
      epochs_no_improve += 1
      print(f'  -> no improvement ({epochs_no_improve}/{patience})')
      if epochs_no_improve >= patience:
        print(f'Early stopping triggered at epoch {epoch+1}')
        break

  # 학습 시간 측정
  end_time = time.time() - start_time
  print('Total time: {}s'.format(end_time))

  # 가장 좋은 모델 가중치로 복원 (early stopping이 없었으면 마지막 epoch 그대로)
  if best_state_dict is not None:
    model.load_state_dict(best_state_dict)

  # 학습된 모델 파라미터 저장
  torch.save(model.state_dict(), MODEL_PATH)
  
  # Loss 그래프를 만들기 위한 히스토리 저장
  with open(HISTORY_PATH, 'wb') as f:
    pickle.dump(history, f)