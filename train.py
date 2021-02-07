import os
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from model import ResNet18
from data import Dataset
import matplotlib
import random

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle

N_EPOCH = 120  # 训练次数
LR = 0.001  # 初始学习率
GAMA = 0.5  # 见LR_DECAY
LR_DECAY = [20, 40, 60,80]  # 学习率变化位置，即 第几个EPOCH进行学习率变化, 变化行为：学习率衰减为上一次的GAMA倍
BATCH_SIZE = 16  # 学习批量
IS_TEST = False  # 如果设置为TRUE，则不进行训练，直接加载chekpoint中的最优模型进行测试
CUDA = True


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    print("===> 建立模型")
    model = ResNet18()  # 模型
    if CUDA: model = model.cuda()
    if IS_TEST:
        model = torch.load("./checkpoint/model_best.pth")
    criterion = nn.BCELoss()  # 损失函数reduction='sum'
    print("===> 加载数据集")
    full_dataset = Dataset(model=1)
    train_set, val_set = torch.utils.data.random_split(full_dataset, [2400, len(full_dataset) - 2400])
    # train_set = Dataset()
    train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=BATCH_SIZE, shuffle=False)
    # val_set=Dataset(istrain=False)
    val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=BATCH_SIZE, shuffle=False)
    print("===> 设置 优化器")
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = lr_scheduler.MultiStepLR(optimizer, LR_DECAY, gamma=GAMA)
    print("===> 进行训练")
    best_loss = 1e8
    loss_plt = []
    acc_plt = []
    if not IS_TEST:
        for epoch in range(N_EPOCH):
            loss_sum = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input, target = batch[0].float(), batch[-1].float()
                if CUDA: input, target = input.cuda(), target.cuda()
                predict = model(input)
                loss = criterion(predict, target)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            loss_plt.append(loss_sum)
            plot(loss_plt)
            print("Epoch:", epoch, "Loss:", loss_sum, "LR:", optimizer.param_groups[0]["lr"])
            if loss_sum < best_loss:
                best_loss = loss_sum
                save_checkpoint(model, epoch)
            scheduler.step()
            torch.set_grad_enabled(False)
            error_num = 0
            sum = 0
            for batch in val_loader:
                input, target = batch[0].float(), batch[-1].float()
                if CUDA: input, target = input.cuda(), target.cuda()
                sum += input.size(0)
                predict = model(input)
                predict[predict < 0.5] = 0
                predict[predict >= 0.5] = 1
                error_num += (predict - target).abs().sum().cpu()
            acc_plt.append(1 - float(error_num) / sum)
            plot2(acc_plt)
            print("Epoch:", epoch, "正确率:", 1 - float(error_num) / sum)
            torch.set_grad_enabled(True)
            with open("./loss_plt.pt", "wb") as f:
                pickle.dump(loss_plt, f)
            with open("./acc_plt.pt", "wb") as f:
                pickle.dump(acc_plt, f)

    else:
        print("===> 跳过训练")
        torch.set_grad_enabled(False)
        error_num = 0
        sum = 0
        for batch in val_loader:
            input, target = batch[0].float(), batch[-1].float()
            if CUDA: input, target = input.cuda(), target.cuda()
            sum += input.size(0)
            predict = model(input)
            predict[predict < 0.5] = 0
            predict[predict >= 0.5] = 1
            error_num += (predict - target).abs().sum().cpu()
        print("===>测试完成", "正确率:", 1 - float(error_num) / sum)
        torch.set_grad_enabled(True)


def save_checkpoint(model, epoch):
    if epoch % 10 == 0:
        model_out_path = "./checkpoint/" + "model_best.pth"
        if not os.path.exists("./checkpoint/"):
            os.makedirs("./checkpoint/")
        torch.save(model, model_out_path)


def plot(data):
    epoch = len(data)
    np.savetxt('log.txt', data, fmt='%0.8f')
    axis = np.linspace(1, epoch, epoch)
    label = 'LOSS PICTURE'
    fig = plt.figure()
    plt.title(label)
    plt.plot(
        axis,
        data,
        label='LOSS_EPOCH {}'.format(epoch)
    )
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.grid(True)
    plt.savefig("./jilu.pdf")
    plt.close(fig)


def plot2(data):
    epoch = len(data)
    np.savetxt('acc.txt', data, fmt='%0.8f')
    axis = np.linspace(1, epoch, epoch)
    label = 'ACC PICTURE'
    fig = plt.figure()
    plt.title(label)
    plt.plot(
        axis,
        data,
        label='ACC_EPOCH'.format(epoch)
    )
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Training accuracy')
    plt.grid(True)
    plt.savefig("./acc.pdf")
    plt.close(fig)


if __name__ == "__main__":
    # 设置随机数种子
    setup_seed(0)
    main()