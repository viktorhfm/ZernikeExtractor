import json
import matplotlib.pyplot as plt
import numpy as np

def normaliza(A):
    mask = np.isnan(A)
    B = np.nan_to_num(A, nan=0)
    C = (B - B.min())/(B.max() - B.min()) * 2 - 1
    C[mask] = 0.0
    return C

def normaliza_pos(A):
    mask = np.isnan(A)
    B = np.nan_to_num(A, nan=0)
    C = (B - B.min())/(B.max() - B.min())
    C[mask] = 0.0
    return C

def generate_mask(reference):
    mask = np.isnan(reference)
    mask = np.logical_not(mask)
    return mask

def plot_graphs_ae(record, epoch):
    x = range(epoch)
    fig, ax1 = plt.subplots(figsize=(10,4))
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('autoencoder_loss')
    ax1.plot(x, record['autoencoder_loss'], label='training_loss', color='tab:red')
    ax1.plot(x, record['validation_loss'], label='validation_loss', color='tab:blue')
    ax1.tick_params(axis='y')

    fig.tight_layout()
    plt.show()


def plot_graphs_zae(record, epoch):
    x = range(epoch)
    fig, ax1 = plt.subplots(figsize=(10,4))
    color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('total_zernike_decoder_loss', color=color)
    ax1.plot(x, record['total_zernike_decoder_loss'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('training time', color=color)
    ax2.plot(x, record['time'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()

def json_to_dict(path, destination_dict):
    loaded_dict = json.load(open(path, 'r'))
    for key in loaded_dict.keys():
        destination_dict[key] = loaded_dict[key]