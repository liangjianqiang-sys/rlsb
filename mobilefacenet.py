"""
MobileFaceNet 人脸特征提取模型
基于 MobileNetV2 的轻量级人脸识别网络，包含：
- ConvBlock: 基础卷积块
- DepthWise: 深度可分离卷积瓶颈块
- Residual: 残差瓶颈块序列
- LinearBlock: 线性全局深度卷积块
- ArcNet: ArcFace 加性角边距损失
"""
import argparse
import functools
import math
import os
import re
import time
from datetime import datetime, timedelta

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

from tool import add_arguments, print_arguments

# ============================================================
# 命令行参数
# ============================================================
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('train_root_path', str, 'dataset/train', '训练数据集根目录')
add_arg('test_root_path', str, 'dataset/test', '测试数据集根目录')
add_arg('batch_size', int, 64, '批次大小')
add_arg('num_epoch', int, 50, '训练轮数')
add_arg('learning_rate', float, 0.001, '初始学习率')
add_arg('num_workers', int, 4, '数据加载线程数')
add_arg('gpus', str, '0', '使用的GPU编号，多个用逗号分隔')
add_arg('resume', str, '', '恢复训练的模型路径')
add_arg('save_model_path', str, 'save_model/', '模型保存目录')
add_arg('input_size', int, 112, '输入图像尺寸')
args = parser.parse_args()


# ============================================================
# 基础模块定义
# ============================================================

class ConvBlock(nn.Module):
    """卷积 + 批归一化 + PReLU 激活"""
    def __init__(self, in_channels, out_channels, kernel, stride, padding, groups=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                              stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class DepthWise(nn.Module):
    """
    深度可分离卷积瓶颈块 (MobileNetV2 风格)
    结构: 1x1扩展 → 3x3深度卷积 → 1x1投影
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel, stride, padding: 深度卷积参数
        groups: 扩展后的通道数（中间层）
    """
    def __init__(self, in_channels, out_channels, kernel, stride, padding, groups):
        super(DepthWise, self).__init__()
        # 1x1 逐点卷积扩展
        self.conv1 = ConvBlock(in_channels, groups, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        # 3x3 深度卷积
        self.conv2 = ConvBlock(groups, groups, kernel=kernel, stride=stride,
                               padding=padding, groups=groups)
        # 1x1 逐点卷积投影
        self.conv3 = nn.Conv2d(groups, out_channels, kernel_size=(1, 1), stride=(1, 1),
                               padding=(0, 0), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn(x)
        return x


class Residual(nn.Module):
    """
    残差瓶颈块序列
    参数:
        in_channels: 输入/输出通道数
        num_block: 瓶颈块数量
        groups: 中间扩展通道数
        kernel, stride, padding: 深度卷积参数
    """
    def __init__(self, in_channels, num_block, groups, kernel, stride, padding):
        super(Residual, self).__init__()
        modules = []
        for i in range(num_block):
            modules.append(ResidualBlock(in_channels, groups, kernel, stride, padding))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    """
    单个残差瓶颈块
    结构: 1x1扩展 → 3x3深度卷积 → 1x1投影 [+ 残差连接]
    """
    def __init__(self, in_channels, groups, kernel, stride, padding):
        super(ResidualBlock, self).__init__()
        # 1x1 扩展
        self.conv1 = ConvBlock(in_channels, groups, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        # 3x3 深度卷积
        self.conv2 = ConvBlock(groups, groups, kernel=kernel, stride=stride,
                               padding=padding, groups=groups)
        # 1x1 投影回原通道数
        self.conv3 = nn.Conv2d(groups, in_channels, kernel_size=(1, 1), stride=(1, 1),
                               padding=(0, 0), bias=False)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn(out)
        out = out + identity
        return out


class LinearBlock(nn.Module):
    """
    线性全局深度卷积块 (替代全局平均池化)
    结构: 7x7深度卷积 → 1x1逐点卷积
    """
    def __init__(self, in_channels, out_channels, groups, kernel, stride, padding):
        super(LinearBlock, self).__init__()
        # 深度卷积
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel,
                               stride=stride, padding=padding, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        # 1x1 逐点卷积
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1),
                               stride=(1, 1), padding=(0, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu(x)
        return x


class Flatten(nn.Module):
    """展平层"""
    def forward(self, x):
        return x.view(x.size(0), -1)


# ============================================================
# MobileFaceNet 主模型
# ============================================================

class MobileFaceNet(nn.Module):
    """
    MobileFaceNet: 轻量级人脸识别网络
    输入: [B, 3, 112, 112]
    输出: [B, 512] 人脸特征嵌入向量
    """
    def __init__(self):
        super(MobileFaceNet, self).__init__()
        # 阶段1: 标准卷积下采样
        self.conv1 = ConvBlock(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        # 阶段2: 深度可分离卷积
        self.conv2_dw = ConvBlock(64, 64, kernel=(3, 3), stride=(1, 1),
                                  padding=(1, 1), groups=64)
        # 阶段2→3: 下采样瓶颈
        self.conv_23 = DepthWise(64, 64, kernel=(3, 3), stride=(2, 2),
                                 padding=(1, 1), groups=128)
        # 阶段3: 4个残差瓶颈块
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3),
                               stride=(1, 1), padding=(1, 1))
        # 阶段3→4: 下采样瓶颈
        self.conv_34 = DepthWise(64, 128, kernel=(3, 3), stride=(2, 2),
                                 padding=(1, 1), groups=256)
        # 阶段4: 6个残差瓶颈块
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3),
                               stride=(1, 1), padding=(1, 1))
        # 阶段4→5: 下采样瓶颈
        self.conv_45 = DepthWise(128, 128, kernel=(3, 3), stride=(2, 2),
                                 padding=(1, 1), groups=512)
        # 阶段5: 2个残差瓶颈块
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3),
                               stride=(1, 1), padding=(1, 1))
        # 分离卷积扩维
        self.conv_6_sep = ConvBlock(128, 512, kernel=(1, 1), stride=(1, 1),
                                    padding=(0, 0))
        # 线性全局深度卷积 (替代池化)
        self.conv_6_dw = LinearBlock(512, 512, groups=512, kernel=(7, 7),
                                     stride=(1, 1), padding=(0, 0))
        # 嵌入层
        self.flatten = Flatten()
        self.linear = nn.Linear(512, 512, bias=False)
        self.bn = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_34(x)
        x = self.conv_4(x)
        x = self.conv_45(x)
        x = self.conv_5(x)
        x = self.conv_6_sep(x)
        x = self.conv_6_dw(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return x


# ============================================================
# ArcFace 损失函数
# ============================================================

class ArcNet(nn.Module):
    """
    ArcFace 加性角边距损失层
    参数:
        in_features: 特征维度
        num_classes: 类别数量
        s: 缩放因子 (默认64.0)
        m: 角边距 (默认0.5弧度)
    """
    def __init__(self, in_features, num_classes, s=64.0, m=0.5):
        super(ArcNet, self).__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        # 预计算三角函数值
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features, labels):
        # L2 归一化
        features = F.normalize(features)
        weight = F.normalize(self.weight)

        # 计算余弦相似度 cos(theta)
        cosine = F.linear(features, weight)

        # 计算 sin(theta)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(1e-12, 1.0))

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # 当 cos(theta) > cos(pi - m) 时使用 phi，否则使用 cos(theta) - m*sin(m)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot 编码
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # 对正确的类别加边距，其余保持原值
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


# ============================================================
# 数据集定义
# ============================================================

class FaceDataset(Dataset):
    """
    人脸数据集
    目录结构: root_path/类别名/图片.jpg
    每个子文件夹为一个类别(人)
    """
    def __init__(self, root_path, is_train=True, image_size=112):
        super(FaceDataset, self).__init__()
        self.image_size = image_size
        self.is_train = is_train

        # 扫描类别目录
        self.classes = sorted(os.listdir(root_path))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        # 收集所有图片路径和标签
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_path, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                self.samples.append((img_path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 读取图像
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # BGR 转 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 缩放到指定尺寸
        img = cv2.resize(img, (self.image_size, self.image_size))

        # 归一化到 [-1, 1]
        img = (img.astype(np.float32) - 127.5) / 127.5

        # 转为 CHW 格式
        img = img.transpose((2, 0, 1))

        return torch.FloatTensor(img), label


# ============================================================
# 辅助函数
# ============================================================

def save_model(args, model, metric_fc, optimizer, epoch_id):
    """保存模型检查点"""
    save_dir = os.path.join(args.save_model_path, 'epoch_%d' % epoch_id)
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, 'model_params.pth'))
    torch.save(metric_fc.state_dict(), os.path.join(save_dir, 'metric_fc_params.pth'))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, 'optimizer.pth'))

    # 同时导出 TorchScript 格式供预测使用
    model.eval()
    dummy_input = torch.randn(1, 3, args.input_size, args.input_size,
                              device=next(model.parameters()).device)
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(os.path.join(args.save_model_path, 'mobilefacenet.pth'))
    model.train()

    print('[%s] 模型已保存到 %s' % (datetime.now(), save_dir))


def test(args, model):
    """在测试集上评估模型准确率"""
    device = next(model.parameters()).device
    test_dataset = FaceDataset(args.test_root_path, is_train=False, image_size=args.input_size)
    if len(test_dataset) == 0:
        print('测试集为空，跳过评估')
        return 0.0

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    model.eval()
    correct = 0
    total = 0
    all_features = []
    all_labels = []

    with torch.no_grad():
        for data in test_loader:
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()

            # 提取特征
            feature = model(data_input)
            all_features.append(feature)
            all_labels.append(label)

    # 使用最近邻分类评估
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # L2归一化
    all_features = F.normalize(all_features, dim=1)

    # 对每个样本，计算与所有其他样本的余弦相似度
    similarity = torch.mm(all_features, all_features.T)

    for i in range(len(all_features)):
        # 排除自身
        sim = similarity[i].clone()
        sim[i] = -float('inf')
        pred_idx = torch.argmax(sim)
        if all_labels[pred_idx] == all_labels[i]:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


# ============================================================
# 训练主流程
# ============================================================

def train():
    device_ids = [int(i) for i in args.gpus.split(',')]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取训练数据
    train_dataset = FaceDataset(args.train_root_path, is_train=True, image_size=args.input_size)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size * len(device_ids),
                              shuffle=True,
                              num_workers=args.num_workers)
    print("[%s] 总数据类别为：%d，样本数为：%d" %
          (datetime.now(), train_dataset.num_classes, len(train_dataset)))

    # 创建模型
    model = MobileFaceNet()
    metric_fc = ArcNet(512, train_dataset.num_classes)

    # 多GPU支持
    if len(device_ids) > 1 and torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
        metric_fc = nn.DataParallel(metric_fc, device_ids=device_ids, output_device=device_ids[0])

    model.to(device)
    metric_fc.to(device)

    # 打印模型结构
    print("模型参数量: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))

    # 恢复训练
    last_epoch = 0
    optimizer = torch.optim.SGD([{'params': model.parameters()},
                                  {'params': metric_fc.parameters()}],
                                lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    if args.resume:
        optimizer_state = torch.load(os.path.join(args.resume, 'optimizer.pth'))
        optimizer.load_state_dict(optimizer_state)
        last_epoch = int(re.findall(r"\d+", args.resume)[-1]) + 1
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(
                torch.load(os.path.join(args.resume, 'model_params.pth')))
            metric_fc.module.load_state_dict(
                torch.load(os.path.join(args.resume, 'metric_fc_params.pth')))
        else:
            model.load_state_dict(
                torch.load(os.path.join(args.resume, 'model_params.pth')))
            metric_fc.load_state_dict(
                torch.load(os.path.join(args.resume, 'metric_fc_params.pth')))
        print('成功加载模型参数和优化方法参数')

    # 开始训练
    sum_batch = len(train_loader) * (args.num_epoch - last_epoch)
    for epoch_id in range(last_epoch, args.num_epoch):
        start = time.time()
        epoch_loss = 0.0

        for batch_id, data in enumerate(train_loader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()

            # 前向传播
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 每100个batch打印一次训练信息
            if batch_id % 100 == 0:
                output_np = output.data.cpu().numpy()
                output_np = np.argmax(output_np, axis=1)
                label_np = label.data.cpu().numpy()
                acc = np.mean((output_np == label_np).astype(int))

                elapsed = time.time() - start
                eta_sec = (elapsed / (batch_id + 1)) * (
                    sum_batch - (epoch_id - last_epoch) * len(train_loader) - batch_id)
                eta_str = str(timedelta(seconds=int(eta_sec)))

                print('[%s] Epoch %d, batch %d/%d, loss: %.4f, accuracy: %.4f, lr: %.6f, eta: %s' %
                      (datetime.now(), epoch_id, batch_id, len(train_loader),
                       loss.item(), acc, scheduler.get_last_lr()[0], eta_str))
            start = time.time()

        # 学习率衰减
        scheduler.step()

        # 评估
        model.eval()
        print('=' * 70)
        accuracy = test(args, model)
        model.train()
        print('[%s] Epoch %d 测试准确率: %.5f' % (datetime.now(), epoch_id, accuracy))
        print('=' * 70)

        # 保存模型
        if len(device_ids) > 1 and torch.cuda.is_available():
            save_model(args, model.module, metric_fc.module, optimizer, epoch_id)
        else:
            save_model(args, model, metric_fc, optimizer, epoch_id)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print_arguments(args)
    train()
