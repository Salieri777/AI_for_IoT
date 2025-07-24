# utils.py

import sys
import librosa
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim


########################
# TRANSFORM DATA TOOLS #
########################


def spec_to_img(spec, eps=1e-6):
    '''
    transform spectrum data into an image
    '''
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    return spec_scaled.astype(np.uint8)


def melspectrogram_db(fpath, sr=None, n_fft=2048, hop_length=512,
                      n_mels=128, fmin=20, fmax=8300, top_db=80):
    '''
    extract mel spectrogram in dB from audio
    '''
    wav, sr = librosa.load(fpath, sr=sr)
    if wav.shape[0] < 5 * sr:
        wav = np.pad(
            wav,
            int(np.ceil((5 * sr - wav.shape[0]) / 2)),
            mode='reflect'
        )
    else:
        wav = wav[:5 * sr]
    spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft,
                                          hop_length=hop_length, n_mels=n_mels,
                                          fmin=fmin, fmax=fmax)
    return librosa.power_to_db(spec, top_db=top_db)


#############################
# NEURAL NET TRAINING TOOLS #
#############################


def set_learning_rate(optimizer, lr):
    '''
    set learning rate to optimizer's parameters
    '''
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def learning_rate_decay(optimizer, epoch, learning_rate):
    '''
    decay learning rate every 20 epochs
    '''
    if epoch % 20 == 0:
        lr = learning_rate / (100 ** (epoch // 20))
        optimizer = set_learning_rate(optimizer, lr)
        print(f'[+] Changing Learning Rate to: {lr}')
    return optimizer


def train(model, train_loader, valid_loader,
          epochs=100, learning_rate=2e-4,
          decay=False, patience=10):
    '''
    training loop with early stopping:
    stops if valid_loss does not improve for `patience` consecutive epochs
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_valid_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for e in range(1, epochs + 1):
        # -- training phase --
        model.train()
        train_losses = []
        if decay:
            optimizer = learning_rate_decay(optimizer, e, learning_rate)

        for x, y in train_loader:
            optimizer.zero_grad()
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)

            preds = model(x)
            loss = loss_func(preds, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        print(f'Epoch {e} Train Loss: {avg_train_loss:.4f}')

        # -- validation phase --
        model.eval()
        valid_losses = []
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x, y in valid_loader:
                x = x.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.long)

                preds = model(x)
                loss = loss_func(preds, y)
                valid_losses.append(loss.item())

                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        avg_valid_loss = np.mean(valid_losses)
        preds_arr = np.concatenate(all_preds, axis=0)
        targets_arr = np.concatenate(all_targets, axis=0)
        valid_acc = (preds_arr.argmax(axis=1) == targets_arr).mean()

        print(f'Epoch {e} Valid Loss: {avg_valid_loss:.4f} | Valid Acc: {valid_acc:.4f}')

        # -- early stopping check --
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f'>>> No improvement for {epochs_no_improve}/{patience} epochs.')
            if epochs_no_improve >= patience:
                print(f'>>> Early stopping at epoch {e}.')
                break

    # load best state before returning
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


#################
# STORING TOOLS #
#################


def save_model(model, fname):
    '''
    save pytorch trained model
    '''
    torch.save(model, fname)


def save_cat_idx(data, fname):
    '''
    save categorical look up tables
    '''
    with open(fname, 'wb') as f:
        pickle.dump(data.idx2cat, f)
