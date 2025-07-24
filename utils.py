
import sys
import librosa
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import pickle

########################
# TRANSFORM DATA TOOLS #
########################

def spec_to_img(spec, eps=1e-6):
    '''
    将梅尔频谱图标准化并转换为图像格式
    '''
    mean = spec.mean()  # 计算均值
    std = spec.std()    # 计算标准差
    spec_norm = (spec - mean) / (std + eps)  # 标准化
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    # 缩放至 0~255，转换为 uint8 图像格式
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled

def melspectrogram_db(fpath, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
    '''
    提取音频文件的梅尔频谱（单位：分贝）
    '''
    wav, sr = librosa.load(fpath, sr=sr)  # 加载音频
    if wav.shape[0] < 5 * sr:
        # 小于5秒则对称补齐
        wav = np.pad(wav, int(np.ceil((5*sr - wav.shape[0])/2)), mode='reflect')
    else:
        # 超过5秒则裁剪
        wav = wav[:5*sr]

    # 计算梅尔频谱并转换为分贝单位
    spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft,
                                          hop_length=hop_length, n_mels=n_mels,
                                          fmin=fmin, fmax=fmax)
    spec_db = librosa.power_to_db(spec, top_db=top_db)
    return spec_db

#############################
# NEURAL NET TRAINING TOOLS #
#############################

def set_learning_rate(optimizer, lr):
    '''
    设置优化器的学习率
    '''
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def learning_rate_decay(optimizer, epoch, learning_rate):
    '''
    每20轮降低一次学习率
    '''
    if epoch % 20 == 0:
        lr = learning_rate / (100 ** (epoch // 20))
        opt = set_learning_rate(optimizer, lr)
        print(f'[+] Changing Learning Rate to: {lr}')
        return opt
    else:
        return optimizer

def train(model, train_loader, valid_loader, epochs=100, learning_rate=2e-5, decay=True):
    '''
    模型训练主循环
    '''
    # 判断是否使用GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    loss_func = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    valid_losses = []

    for e in range(1, epochs + 1):
        model.train()
        batch_losses = []

        if decay:
            opt = learning_rate_decay(opt, e, learning_rate)

        for i, data in enumerate(train_loader):
            x, y = data
            opt.zero_grad()
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)

            # 前向传播
            preds = model(x)
            loss = loss_func(preds, y)
            loss.backward()
            batch_losses.append(loss.item())
            opt.step()

            # 打印进度条（美化用）
            bar = '-' * (i // 4)
            sys.stdout.write('[i] FORWARD' + bar + '> * <' + bar + 'BACKWARD [i] ')

        train_losses.append(batch_losses)
        print(f'Epoch - {e} Train-Loss: {np.mean(train_losses[-1]):.4f}')

        # 验证阶段
        model.eval()
        batch_losses = []
        trace_y = []
        trace_pred = []

        for i, data in enumerate(valid_loader):
            x, y = data
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            preds = model(x)
            loss = loss_func(preds, y)
            batch_losses.append(loss.item())
            trace_y.append(y.cpu().detach().numpy())
            trace_pred.append(preds.cpu().detach().numpy())

            bar = '-' * (i // 4)
            sys.stdout.write('[i] FORWARD' + bar + '> *  ')

        valid_losses.append(batch_losses)
        trace_y = np.concatenate(trace_y)
        trace_pred = np.concatenate(trace_pred)
        acc = np.mean(trace_pred.argmax(axis=1) == trace_y)

        print(f'Epoch - {e} Valid-Loss: {np.mean(valid_losses[-1]):.4f} Valid Accuracy: {acc:.4f}')

    return model

#################
# STORING TOOLS #
#################

def save_model(model, fname):
    '''
    保存 PyTorch 模型
    '''
    with open(fname, 'wb') as f:
        torch.save(model, f)

def save_cat_idx(data, fname):
    '''
    保存类别索引映射字典（idx2cat）
    '''
    with open(fname, 'wb') as f:
        pickle.dump(data.idx2cat, f)