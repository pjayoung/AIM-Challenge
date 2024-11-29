import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import trange
from Age_Estimation.model2 import MultipleOutputCNN
from Age_Estimation.utils import *
from dataset.dataset import CustomDataset

'''
Hyper-parameters
'''
save_path = '/raid/co_show02/JY/CNN/5train'
epoch = 1000  # 200
learning_rate = 1e-2
batch_size = 1024
split = np.array([7, 2, 1])
shuffle_dataset = True
random_seed = 42
np.random.seed(random_seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
l2_loss_fn = torch.nn.CrossEntropyLoss()  # 범주형 분류를 위한 손실 함수로 변경
patience = 10  # Number of epochs to wait for improvement before stopping
early_stopping_counter = 0

'''
Load full dataset
'''

# 학습 코드에서 사용할 데이터셋 경로와 데이터셋 클래스 변경
label_file = '/raid/co_show02/JY/CNN/txt/ca_rawlabels.txt'  # 사용자가 정의한 라벨 파일 경로
full_dataset = CustomDataset(label_file=label_file)
print('데이터셋 로드')

# 학습 코드에서 기존 AFAD 대신 CustomDataset으로 변경
train_dataset, test_dataset, val_dataset = split_dataset(full_dataset=full_dataset, split=split, is_val=True)
train_dataset.dataset.is_train = True
train_dataset.dataset.Image_Transform()
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
)
if val_dataset:
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
print('완료')


def val_loop(model, loader, device):
    total = len(loader.dataset)
    mae = 0
    for step, batch in enumerate(loader):
        x, label, age = batch
        x, label = x.double(), label.double()
        x = x.to(device)
        label = label.to(device)
        age = age.to(device)
        predict = model(x)
        mae += MAE(predict, age) * len(age)
    mae = mae / total
    print('validate|| MAE:{:.5f}'.format(mae))
    return mae


def train_loop(model, loader, optimizer, loss_func, device, importance):
    total = len(loader.dataset)
    importance = importance.to(device)
    for step, batch in enumerate(loader):
        x, label, age = batch
        x, label = x.double(), label.double()
        x = x.to(device)
        label = label.to(device)
        age = age.to(device)
        predict = model(x)
        loss = loss_func(predict, label, importance).to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mae = MAE(predict, age)
        print('training || loss:{:.7f} MAE:{:.5f} [{}/{}]'.format(loss.item(), mae, len(x) * (step + 1), total))
    pass

def raw_save_model(model, save_path, path, is_best):
    torch.save(model.state_dict(), save_path + path)
    if is_best:
        torch.save(model.state_dict(), save_path + '_best.pth')

def main():
    global early_stopping_counter
    model = MultipleOutputCNN().double()
    model.to(device)
    print(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.5, last_epoch=-1, verbose=False)
    best_MAE = float('inf')
    is_best = 0
    importance = make_task_importance()
    for i in trange(epoch):
        print('-----------------------epoch {}-----------------------'.format(i + 1))
        print('-----------current learning rate: {:.6f}-----------'.format(
            optimizer.state_dict()['param_groups'][0]['lr']))
        model.train()
        train_loop(model, train_dataloader, optimizer, importance_cross_entropy, device, importance)
        with torch.no_grad():
            model.eval()
            mae = val_loop(model, val_dataloader, device)
        if mae < best_MAE:
            best_MAE = mae
            is_best = 1
            early_stopping_counter = 0  # Reset the counter if there is an improvement
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {i + 1} epochs.")
            break

        raw_save_model(model, save_path, '5_epoch_{}.pth'.format(i + 1), is_best)
        scheduler.step()
        is_best = 0
    pass


if __name__ == '__main__':
    main()
