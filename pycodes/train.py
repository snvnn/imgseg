import torch
from copy import deepcopy
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
import time
import pickle
from configuration import LEARNING_RATE, EPOCHS,HISTORY_PATH, DEVICE, MODEL_PATH, MIN_DELTA, BEST_LOSS, PATIENCE

CHECKPOINT_PATH = MODEL_PATH + ".ckpt"


def train(model, train_loader, optimizer=None, scheduler=None, start_epoch=0):

  model.train()

  loss_fn = CrossEntropyLoss()
  # 외부에서 optimizer를 넘겨주지 않으면 기본 Adam 생성
  if optimizer is None:
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

  # 외부에서 scheduler를 넘겨주지 않으면 기본 ReduceLROnPlateau 생성
  if scheduler is None:
    scheduler = ReduceLROnPlateau(
      optimizer,
      mode="min",
      factor=0.5,
      patience=3,
      min_lr=1e-6,
      verbose=True
    )

  history = {'train_loss': []}
  start_time = time.time()

  # Early stopping 설정
  patience = PATIENCE         # 개선이 없을 때 허용할 최대 epoch 수
  min_delta = MIN_DELTA       # 개선으로 인정할 최소 손실 감소량
  best_loss = BEST_LOSS

  best_state_dict = None
  epochs_no_improve = 0

  try:
    for epoch in range(start_epoch, EPOCHS):
      print('EPOCH: {}'.format(epoch))
      train_loss = 0.

      for i, (images, targets) in enumerate(train_loader):
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        pred = model(images)
        loss = loss_fn(pred, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss*len(images)

      avg_train_loss = train_loss/len(train_loader.dataset)
      print('TRAIN LOSS: {}'.format(avg_train_loss))
      history['train_loss'].append(avg_train_loss.cpu().detach().numpy())

      # 현재 epoch의 loss 값
      current_loss = avg_train_loss.item()

      # LR 스케줄러 업데이트 (plateau 기준)
      if scheduler is not None:
        scheduler.step(current_loss)

      # Early stopping 로직 (train loss 기준)
      if best_loss - current_loss > min_delta:
        best_loss = current_loss
        best_state_dict = deepcopy(model.state_dict())
        epochs_no_improve = 0
        print(f'  -> improvement, best_loss updated to {best_loss:.6f}')
        # 개선될 때마다 체크포인트 저장 (중간에 중단되어도 유지)
        torch.save(best_state_dict, MODEL_PATH)
        print(f'  -> checkpoint saved to {MODEL_PATH}')
      else:
        epochs_no_improve += 1
        print(f'  -> no improvement ({epochs_no_improve}/{patience})')
        if (patience > 0 and epochs_no_improve >= patience):
          print(f'Early stopping triggered at epoch {epoch+1}')
          break
        
      # 매 epoch 끝날 때 체크포인트 저장 (resume 용)
      checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None
      }
      torch.save(checkpoint, CHECKPOINT_PATH)
    
  except KeyboardInterrupt:
    # 학습 중 사용자가 중단한 경우에도 현재까지의 최고 모델을 저장
    print('Training interrupted by user. Saving checkpoint...')
    if best_state_dict is not None:
      torch.save(best_state_dict, MODEL_PATH)
      print(f'Saved best model so far to {MODEL_PATH}')
    else:
      torch.save(model.state_dict(), MODEL_PATH)
      print(f'Saved current model to {MODEL_PATH}')

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