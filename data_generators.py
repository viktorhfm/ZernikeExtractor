import random
import numpy as np
import tensorflow as tf
from utils import normaliza_pos
import matplotlib.pyplot as plt
from zernike import RZern


def GeneradorZ(cart, num_samples=50000, num_coef=10, h=256, w=256):
    y = np.empty((num_samples, num_coef))
    X = np.empty((num_samples, h, w))
    Xcos = np.empty((num_samples, h, w))
    for k in range(num_samples):
        y[k] = [
            random.random() * 20 - 10,
            random.random() * 30 - 15,
            random.random() * 30 - 15,
            random.random() * 30 - 15,
            random.random() * 30 - 15,
            random.random() * 30 - 15
        ]
        Phi = cart.eval_grid(y[k], matrix=True)
        X[k,:,:] = normaliza_pos(Phi)
        Xcos[k,:,:] = normaliza_pos(np.cos(Phi))
    return X, y, Xcos

def GeneradorZ_sparse(cart, num_samples=50000, num_coef=10, h=256, w=256):
    y = np.empty((num_samples,num_coef))
    X = np.empty((num_samples, h, w))
    Xcos = np.empty((num_samples, h, w))
    for k in range(num_samples):
        y[k] = np.zeros(num_coef)
        n_terms = np.random.randint(1,num_coef)
        for cont in range(n_terms):
            index = np.random.randint(num_coef)
            if index == 0:
                y[k, index] = random.random() * 20 - 10
            else:
                y[k, index] = random.random() * 30 - 15
        Phi = cart.eval_grid(y[k], matrix=True)
        X[k,:,:] = normaliza_pos(Phi)
        Xcos[k,:,:] = normaliza_pos(np.cos(Phi))
    return X, y, Xcos

def generate_zernike(coef, cart, normalize=True):
    Phi = cart.eval_grid(coef, matrix=True)
    if normalize:
        Phi = normaliza_pos(Phi)
    return Phi

def zernike2phi(coef):
    # Phi = tf.map_fn(generate_zernike, coef)
    Phi = tf.map_fn(
        lambda x: generate_zernike(x, normalize=False),
        coef
    )
    B = np.nan_to_num(Phi, nan=0)
    return B

def zernike2cos(coef):
    Phi = tf.map_fn(
        lambda x: generate_zernike(x, normalize=False),
        coef
    )
    Phi = tf.math.cos(Phi)
    B = np.nan_to_num(Phi, nan=0)
    return B

def zernike2gradient(coef):
    Phi = tf.map_fn(
        lambda x: generate_zernike(x, normalize=False),
        coef
    )
    Phi = tf.convert_to_tensor(
        np.expand_dims(
            np.nan_to_num(Phi), 
            axis=3
        )
    )
    dx, dy = tf.image.image_gradients(Phi)
    return dx, dy

def generate_images(models, test_input, tar, cart): #JV

    encoder = models[0]
    decoder = models[1]
    zernike_decoder = models[2]

    encoded_image = encoder(test_input, training=True)
    prediction = decoder(encoded_image, training=True)
    zernikes = zernike_decoder(encoded_image, training=True)

    generated_zernike = np.nan_to_num(
        generate_zernike(zernikes[0].numpy(), cart, normalize=False), #JV 
        nan=0
    )
    
    got = np.nan_to_num(
        generate_zernike(tar[0].numpy(), cart, normalize=False), #JV
        nan=0
    )
    error = np.abs(got - generated_zernike)

    plt.figure(figsize=(25,15))

    display_list = [test_input[0], prediction[0], generated_zernike, got, error]
    title = [
        'Input Image', 
        'Autoencoder', 
        np.round(zernikes[0].numpy(), decimals=2), 
        np.round(tar[0].numpy(), decimals=2),
        'Phase Error'
    ]
    
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i])
        plt.colorbar(fraction=0.046)
        plt.axis('off')
    #plt.show()

def generate_images_exp(models, test_input, tar, save=False):

    encoder = models[0]
    decoder = models[1]
    zernike_decoder = models[2]

    encoded_image = encoder(test_input, training=True)
    prediction = normaliza_pos(decoder(encoded_image, training=True))
    zernikes = zernike_decoder(encoded_image, training=True)
    
    got = np.nan_to_num(
        normaliza_pos(generate_zernike(tar[0].numpy(), normalize=False)),
        nan=0
    )
    
    generated_zernike = np.nan_to_num(
        normaliza_pos(generate_zernike(zernikes[0].numpy(), normalize=False)), 
        nan=0
    )
    error = np.sqrt(np.abs(got - generated_zernike))

    plt.figure(figsize=(25,15))

    display_list = [
        normaliza_pos(test_input[0]), 
        normaliza_pos(prediction[0]), 
        got, 
        generated_zernike, 
        error
    ]

    title = [
        r'a) $I_{exp}$', 
        r'b) $\hat I$', 
        # np.round(tar[0].numpy(), decimals=2),
        # np.round(zernikes[0].numpy(), decimals=2), 
        r'c) $\phi$',
        r'd) $\hat \phi$',
        r'e) RMSE'
    ]
    
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.title(title[i], fontdict={'fontsize': 20}, y=-0.15)
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i])
        plt.colorbar(fraction=0.046)
        plt.axis('off')
    if save:
        plt.savefig('./images/test_result.png', bbox_inches='tight')
    #plt.show()
