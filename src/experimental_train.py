from data_generators import *
from IPython import display
from loss_functions import *
from sklearn.utils import shuffle
from tensorflow.python.ops.numpy_ops import np_config
from utils import *
from zernike import RZern
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tensorflow as tf
import time

def swap_columns(arr, frm, to):
    arr[:,[frm, to]] = arr[:,[to, frm]]

def model_compiler(path, optimizer, loss):
    model = tf.keras.models.load_model(path, compile=False)
    model.compile(optimizer=optimizer, 
                  loss=loss)
    return model

# Fine tuning Functions

@tf.function()
def train_step_zae(
    input_image, 
    target, 
    encoder,
    zernike_decoder,
    zernike_decoder_optimizer
):
    with tf.GradientTape() as zernike_decoder_tape:
        encoded_image = encoder(input_image)
        zernikes = zernike_decoder(encoded_image, training=True)

        # Loss calculation
        zernike_decoder_loss = zernike_loss(target, zernikes)
        estimated_phase_loss = phase_loss(target, zernikes)
        estimated_grad_loss = grad_loss(target, zernikes)

        # Total loss
        total_zernike_decoder_loss = zernike_decoder_loss +\
                                     estimated_phase_loss + \
                                     estimated_grad_loss

    zernike_decoder_gradients = zernike_decoder_tape.gradient(
        total_zernike_decoder_loss, 
        zernike_decoder.trainable_variables
    )
    
    zernike_decoder_optimizer.apply_gradients(zip(
        zernike_decoder_gradients, 
        zernike_decoder.trainable_variables
    )) 

    return [zernike_decoder_loss, 
            estimated_phase_loss, 
            estimated_grad_loss]

@tf.function()
def validation_step_zae(
    input_image, 
    target, 
    encoder, 
    zernike_decoder
):
    encoded_image = encoder(input_image)
    zernikes = zernike_decoder(encoded_image, training=False)
     # Loss calculation
    zernike_decoder_loss = zernike_loss(target, zernikes)
    estimated_phase_loss = phase_loss(target, zernikes)
    estimated_grad_loss = grad_loss(target, zernikes)

    # Total loss
    total_zernike_decoder_loss = zernike_decoder_loss +\
                                    estimated_phase_loss + \
                                    estimated_grad_loss
    return total_zernike_decoder_loss

def plot_graphs_zae(record, epoch):
    x = range(epoch)
    fig, ax1 = plt.subplots(figsize=(10,4))
    color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Training loss', color=color)
    ax1.plot(x, record['total_zernike_decoder_loss_train'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation loss', color=color)
    ax2.plot(x, record['total_zernike_decoder_loss_val'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()


def fit_zae(
    train_ds, 
    val_ds, 
    test_ds, 
    epochs,
    encoder,
    decoder,
    zernike_decoder,
    zernike_decoder_optimizer
):
    tf.compat.v1.enable_eager_execution()

    record = {
        'zernike_decoder_loss': [],
        'estimated_phase_loss': [],
        'estimated_grad_loss': [],
        'total_zernike_decoder_loss_train': [], 
        'total_zernike_decoder_loss_val': [], 
        'time': []
    }

    example_input, example_target = next(iter(test_ds.take(1)))
    
    lowest_loss = 1000
    saved_epoch = 0
    for epoch in range(epochs):

        start = time.time()
        print("Epoch: ", epoch + 1)

        n = 0
        for input_image, target in train_ds:
            zernike_decoder_loss = train_step_zae(
                input_image=input_image,
                target=target,
                encoder=encoder,
                zernike_decoder=zernike_decoder,
                zernike_decoder_optimizer=zernike_decoder_optimizer
            )
            if n % 10 == 0:
                print('··', end='')
            n += 1

        accumulated_loss_val = 0
        for input_image, target in val_ds:
            zernike_decoder_loss_val = validation_step_zae(input_image=input_image, 
                                                           target=target,
                                                           encoder=encoder,
                                                           zernike_decoder=zernike_decoder)
            accumulated_loss_val += zernike_decoder_loss_val.numpy()
        accumulated_loss_val /= len(val_ds)
        
        display.clear_output(wait=True)
        generate_images([encoder, decoder, zernike_decoder], 
                        example_input, 
                        example_target)
        
        record['zernike_decoder_loss'].append(zernike_decoder_loss[0].numpy())
        record['estimated_phase_loss'].append(zernike_decoder_loss[1].numpy())
        record['estimated_grad_loss'].append(zernike_decoder_loss[2].numpy())
        record['total_zernike_decoder_loss_train'].append(
            zernike_decoder_loss[0].numpy() + \
            zernike_decoder_loss[1].numpy() + \
            zernike_decoder_loss[2].numpy()
        )
        record['total_zernike_decoder_loss_val'].append(accumulated_loss_val)
        delta_time = time.time() - start
        record['time'].append(delta_time)

        plot_graphs_zae(record, epoch + 1)

        print("Time taken: ", delta_time)
        print("Zernike Decoder Loss: ", zernike_decoder_loss[0].numpy())
        print("Estimated Phase Loss: ", zernike_decoder_loss[1].numpy())
        print("Estimated Gradient Loss: ", zernike_decoder_loss[2].numpy())
        print("Total Zernike Decoder Loss Training: ", 
              zernike_decoder_loss[0].numpy() + \
              zernike_decoder_loss[1].numpy() + \
              zernike_decoder_loss[2].numpy())
        print("Total Zernike Decoder Loss Validation: ", accumulated_loss_val)
        
        # if epoch > round(epochs * 0.8):
        if record['total_zernike_decoder_loss_train'][-1] < lowest_loss:
            zernike_decoder.save(f'./temp/zae_zernike_decoder_experimental_temp_{epoch}.h5')
            saved_epoch = epoch + 1
            lowest_loss = record['total_zernike_decoder_loss_train'][-1]
        
        print("Saved model on epoch: ", saved_epoch)
        print("Total Zernike Decoder Loss: ", lowest_loss)

    return record

def main():
    HEIGHT = 256
    WIDTH = 256
    cart = RZern(2)
    ddx = np.linspace(-1.0, 1.0, WIDTH)
    ddy = np.linspace(-1.0, 1.0, HEIGHT)
    xv, yv = np.meshgrid(ddx, ddy)
    cart.make_cart_grid(xv, yv)
    num_coef = cart.nk
    print("Number of coefficients: ", num_coef)
    Phi = cart.eval_grid(np.array([0,1,0,0,0,0]), matrix=True)
    MASK = generate_mask(Phi)

    # Load image names without complements
    experimental_path = './Datasets/Zernikes/Train/'
    fnames = os.listdir(experimental_path)
    clean_fnames = [fname for fname in fnames if '_c' not in fname]
    clean_fnames.sort()
    print(f"Loaded {len(clean_fnames)} image names")

    # Load Labels
    path_combinacion = './Datasets/Zernikes/Json/Zernikes_coef_combinacion.json'
    path_puros = './Datasets/Zernikes/Json/Zernikes_coef_puros.json'
    path_random = './Datasets/Zernikes/Json/Zernikes_coef_random.json'
    mega_json = {}
    json_to_dict(path_combinacion, mega_json)
    json_to_dict(path_puros, mega_json)
    json_to_dict(path_random, mega_json)
    print(f"Loaded {len(mega_json)} labels")

    # Load images
    X = []
    y = []
    for fname in clean_fnames:
        X.append(cv2.imread(experimental_path + fname, cv2.IMREAD_GRAYSCALE) / 255.0)
        y.append(mega_json[fname])
    print("Loaded images: ", len(X))
    print("Loaded labels: ", len(y))
    X = np.array(X)
    y = np.array(y)
    print("X shape: ", X.shape) 
    print("y shape: ", y.shape)

    # Swap columns
    swap_columns(y, 1, 2)
    
    # Create Dataset
    NUM_SAMPLES = len(X)

    num_samples_train = round(NUM_SAMPLES * .9)
    X, y = shuffle(X, y)
    X_train = np.expand_dims(X[:num_samples_train], axis=3)
    y_train = y[:num_samples_train]
    X_test = np.expand_dims(X[num_samples_train:], axis=3)
    y_test = y[num_samples_train:]
    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)

    BATCH_SIZE = 64
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(BATCH_SIZE)
    validation_split = 0.1
    num_samples = len(train_dataset)
    num_train = round(num_samples * (1 - validation_split))
    train_ds = train_dataset.take(num_train).batch(BATCH_SIZE)
    val_dataset = train_dataset.skip(num_train).batch(BATCH_SIZE)

    # Load models
    autoencoder_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
    zernike_decoder_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
    encoder = model_compiler('./models/ae_encoder.h5',
                       autoencoder_optimizer,
                       ae_loss)
    decoder = model_compiler('./models/ae_decoder.h5',
                        autoencoder_optimizer,
                        ae_loss)
    zernike_decoder = model_compiler('./models/zae_zernike_decoder.h5',
                                zernike_decoder_optimizer,
                                total_zernike_loss)
    
    np_config.enable_numpy_behavior()
    tf.config.run_functions_eagerly(True)
    EPOCHS = 150
    record = fit_zae(train_ds=train_ds, 
                     val_ds=val_dataset, 
                     test_ds=test_dataset, 
                     epochs=EPOCHS,
                     encoder=encoder,
                     decoder=decoder,
                     zernike_decoder=zernike_decoder,
                     zernike_decoder_optimizer=zernike_decoder_optimizer)
    
    # Evaluate on test data
    np_config.enable_numpy_behavior()
    tf.config.run_functions_eagerly(True)
    zautoencoder = tf.keras.models.Model(
        inputs=encoder.inputs, 
        outputs=zernike_decoder(encoder.outputs)
    )
    zautoencoder.compile(optimizer=zernike_decoder_optimizer, 
                        loss=total_zernike_loss)
    zautoencoder.evaluate(X_test, y_test)
    
    # Save model
    # zernike_decoder.save('./models/zae_zernike_decoder_experimental.h5')

    # Save record
    with open('./records/zae_experimental.pkl', 'wb') as f:
        pickle.dump(record, f)

if __name__ == '__main__':
    main()
    