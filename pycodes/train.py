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

  # 학습 시간 측정
  end_time = time.time() - start_time
  print('Total time: {}s'.format(end_time))
  # 학습된 모델 파라미터 저장
  torch.save(model.state_dict(), MODEL_PATH)
  
    # Loss 그래프를 만들기 위한 히스토리 저장
  with open(HISTORY_PATH, 'wb', ) as f:
      pickle.dump(history, f)