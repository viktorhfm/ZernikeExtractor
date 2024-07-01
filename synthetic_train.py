from data_generators import *
from IPython import display
from loss_functions import *
from sklearn.utils import shuffle
from tensorflow.python.ops.numpy_ops import np_config
from utils import *
from zernike import RZern
import numpy as np
import pickle
import tensorflow as tf
import time


OUTPUT_CHANNELS = 1

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()

    result.add(
        tf.keras.layers.Conv2D(
            filters,
            size,
            # strides=2,
            padding='same',
            kernel_initializer=initializer
            # use_bias=False
        )
    )
    
    result.add(
        tf.keras.layers.Conv2D(
            filters,
            4,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            use_bias=False
        )
    )

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    
    result = tf.keras.Sequential()
    
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            4,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            use_bias=False
        )
    )

    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            size,
            # strides=2,
            padding='same',
            kernel_initializer=initializer
            # use_bias=False
        )
    )

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.LeakyReLU())
        
    return result

def Encoder(attention=False):
    inputs = tf.keras.layers.Input(shape=[256, 256, 1])

    down_stack = [
        downsample(16, 3), # (bs, 128, 128, 16)
        downsample(32, 3), # (bs, 64, 64, 32)
        downsample(64, 3), # (bs, 32, 32, 64)
        downsample(128, 3), # (bs, 16, 16, 128)Max-Pool
        downsample(256, 3), # (bs, 8, 8, 256)
        downsample(256, 3), # (bs, 4, 4, 256)
        downsample(512, 3), # (bs, 2, 2, 512)
        # downsample(512, 3), # (bs, 1, 1, 512)
    ]

    x = inputs

    # Downsampling through the model
    # skips = []
    for down in down_stack:
        x = down(x)
        [height, width, chan] = x.shape[1:]
        if attention:
            # Attention to layers 32 to 8
            if height <= 32 and height >= 8:
                x = tf.reshape(x, [-1, height*width, chan])
                x = tf.keras.layers.Attention()([x, x])
                x = tf.reshape(x, [-1, height, width, chan])
        # skips.append(x)
    latent_space_dim = (x.shape)[1:]
    x = tf.keras.layers.Flatten()(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name='encoder'), latent_space_dim

def Decoder(latent_space_dim):

    inputs = tf.keras.layers.Input(shape=(np.prod(latent_space_dim),))
    inputs_reshaped = tf.keras.layers.Reshape(target_shape=(latent_space_dim))(inputs)

    up_stack =[
        # upsample(512, 3), # (bs, 2, 2, 512)    
        upsample(512, 3), # (bs, 4, 4, 256)
        upsample(256, 3), # (bs, 8, 8, 256)
        upsample(256, 3), # (bs, 16, 16, 128)
        upsample(128, 3), # (bs, 32, 32, 64)
        upsample(64, 3), # (bs, 64, 64, 32)
        upsample(32, 3), # (bs, 128, 128, 16)
    ]

    initializer = tf.random_normal_initializer(0.0, 0.02)

    last = tf.keras.layers.Conv2DTranspose(
        OUTPUT_CHANNELS,
        kernel_size=4,
        strides=2,
        padding='same',
        kernel_initializer=initializer,
        activation='relu'
    )

    x = inputs_reshaped

    for up in up_stack:
        x = up(x)
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name='decoder')

# Define the latent decoder model

def latent_decoder(latent_space_dim):

    dense_input = tf.keras.layers.Input(shape=(np.prod(latent_space_dim),), 
                                        name="dense_input")

    D1 = tf.keras.layers.Dense(np.prod(latent_space_dim))(dense_input)
    D1 = tf.keras.layers.BatchNormalization()(D1)
    D1 = tf.keras.activations.tanh(D1)
    # D1 = tf.keras.layers.LeakyReLU()(D1)

    D2 = tf.keras.layers.Dense(1024)(D1)
    D2 = tf.keras.layers.BatchNormalization()(D2)
    D2 = tf.keras.activations.tanh(D2)
    # D2 = tf.keras.layers.LeakyReLU()(D2)

    D3 = tf.keras.layers.Dense(512)(D2)
    D3 = tf.keras.layers.BatchNormalization()(D3)
    D3 = tf.keras.activations.tanh(D3)
    # D3 = tf.keras.layers.LeakyReLU()(D3)

    D4 = tf.keras.layers.Dense(256)(D3)
    D4 = tf.keras.layers.BatchNormalization()(D4)
    D4 = tf.keras.activations.tanh(D4)
    # D4 = tf.keras.layers.LeakyReLU()(D4)

    D5 = tf.keras.layers.Dense(128)(D4)
    D5 = tf.keras.layers.BatchNormalization()(D5)
    D5 = tf.keras.activations.tanh(D5)
    # D5 = tf.keras.layers.LeakyReLU()(D5)

    D6 = tf.keras.layers.Dense(64)(D5)
    D6 = tf.keras.layers.BatchNormalization()(D6)
    D6 = tf.keras.activations.tanh(D6)
    # D6 = tf.keras.layers.LeakyReLU()(D6)

    D7 = tf.keras.layers.Dense(32)(D6)
    D7 = tf.keras.layers.BatchNormalization()(D7)
    last = tf.keras.layers.Dense(6)(D7)

    return tf.keras.Model(inputs=dense_input, 
                          outputs=last, 
                          name='latent_decoder')

# Define the train step and validation step functions for the Autoencoder

@tf.function()
def train_step_ae(
    input_image, 
    encoder, 
    decoder, 
    autoencoder_optimizer
):
    with tf.GradientTape() as autoencoder_tape:
        encoded_image = encoder(input_image, training=True)
        decoded_image = decoder(encoded_image, training=True)

        autoencoder_loss = ae_loss(input_image, decoded_image)

    autoencoder_gradients = autoencoder_tape.gradient(
        autoencoder_loss, 
        encoder.trainable_variables + decoder.trainable_variables
    )

    autoencoder_optimizer.apply_gradients(zip(
        autoencoder_gradients, 
        encoder.trainable_variables + decoder.trainable_variables
    ))
    
    return autoencoder_loss

@tf.function()
def validation_step_ae(input_image, encoder, decoder):
    encoded_image = encoder(input_image, training=False)
    decoded_image = decoder(encoded_image, training=False)
    autoencoder_loss = ae_loss(input_image, decoded_image)
    return autoencoder_loss

# Define the training loop for the Autoencoder

def fit_ae(
        train_ds, 
        val_ds, 
        test_ds, 
        epochs,
        encoder,
        decoder,
        zernike_decoder,
        cart, #JV
        autoencoder_optimizer
):

    record = {
        'autoencoder_loss': [],
        'validation_loss': []
    }

    example_input, example_target = next(iter(test_ds.take(1)))
    
    for epoch in range(epochs):

        start = time.time()
        print("Epoch: ", epoch)

        n = 0
        for input_image, target in train_ds:
            autoencoder_loss = train_step_ae(input_image,
                                             encoder,
                                             decoder,
                                             autoencoder_optimizer)
            if n % 10 == 0:
                print('.', end='')
            n += 1
        
        for input_image, target in val_ds:
            validation_loss = validation_step_ae(input_image,
                                                 encoder,
                                                 decoder)
    
        display.clear_output(wait=True)
        generate_images([encoder, decoder, zernike_decoder], 
                        example_input, 
                        example_target,
                        cart) #JV
        
        record['autoencoder_loss'].append(autoencoder_loss.numpy())
        record['validation_loss'].append(validation_loss.numpy())
        plot_graphs_ae(record, epoch + 1)

        print("Time taken: ", time.time() - start)
        print("Autoencoder Loss: ", autoencoder_loss.numpy())
        
    return record

# Define the train step for the Zernike Autoencoder

@tf.function()
def train_step_zae(
    input_image, 
    target,
    encoder,
    zernike_decoder,
    zernike_decoder_optimizer,
    cart
): #JV
    with tf.GradientTape() as zernike_decoder_tape:
        encoded_image = encoder(input_image)
        zernikes = zernike_decoder(encoded_image, training=True)

        print("JV2",cart)
        # Loss calculation
        zernike_decoder_loss = zernike_loss(target, zernikes)
        estimated_phase_loss = phase_loss(target, zernikes, cart) #JV
        estimated_grad_loss =  grad_loss(target, zernikes)

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

# Define the training loop for the Zernike Autoencoder

def fit_zae(
    train_ds, 
    test_ds, 
    epochs, 
    encoder,
    decoder,
    zernike_decoder,
    zernike_decoder_optimizer,
    cart
):

                     
    tf.compat.v1.enable_eager_execution()

    record = {
        'zernike_decoder_loss': [],
        'estimated_phase_loss': [],
        'estimated_grad_loss': [],
        'total_zernike_decoder_loss': [], 
        'time': []
    }

    example_input, example_target = next(iter(test_ds.take(1)))
    
    print("JV3",cart)
    for epoch in range(epochs):

        start = time.time()
        print("Epoch: ", epoch + 1)

        n = 0
        for input_image, target in train_ds:
            zernike_decoder_loss = train_step_zae(input_image,
                                                  target,
                                                  encoder,
                                                  zernike_decoder,
                                                  zernike_decoder_optimizer,
                                                  cart) #JV
            if n % 10 == 0:
                print('··', end='')
            n += 1
        
        display.clear_output(wait=True)
        generate_images([encoder, decoder, zernike_decoder], 
                        example_input, 
                        example_target,
                        cart) #JV
        
        record['zernike_decoder_loss'].append(zernike_decoder_loss[0].numpy())
        record['estimated_phase_loss'].append(zernike_decoder_loss[1].numpy())
        record['estimated_grad_loss'].append(zernike_decoder_loss[2].numpy())
        record['total_zernike_decoder_loss'].append(
            zernike_decoder_loss[0].numpy() + \
            zernike_decoder_loss[1].numpy() + \
            zernike_decoder_loss[2].numpy()
        )
        delta_time = time.time() - start
        record['time'].append(delta_time)

        plot_graphs_zae(record, epoch + 1)

        print("Time taken: ", delta_time)
        print("Zernike Decoder Loss: ", zernike_decoder_loss[0].numpy())
        print("Estimated Phase Loss: ", zernike_decoder_loss[1].numpy())
        print("Estimated Gradient Loss: ", zernike_decoder_loss[2].numpy())
        print("Total Zernike Decoder Loss: ", 
              zernike_decoder_loss[0].numpy() + \
              zernike_decoder_loss[1].numpy() + \
              zernike_decoder_loss[2].numpy())

    return record


def main():
    # Create the Zernike object and parameters
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
    NUM_SAMPLES = 11000

    # Generate the synthetic data

    X1, y1, Xcos1 = GeneradorZ(cart=cart,
                            num_samples=NUM_SAMPLES//2, 
                            num_coef=num_coef, 
                            h=HEIGHT, 
                            w=WIDTH)
    X2, y2, Xcos2 = GeneradorZ_sparse(cart=cart,
                                    num_samples=NUM_SAMPLES//2, 
                                    num_coef=num_coef, 
                                    h=HEIGHT, 
                                    w=WIDTH)
    X = np.concatenate((X1, X2))
    y = np.concatenate((y1, y2))
    Xcos = np.concatenate((Xcos1, Xcos2))

    # Shuffle the data and split it into training and testing sets

    num_samples_train = round(NUM_SAMPLES * .9)
    X, y, Xcos= shuffle(X, y, Xcos)
    X_train = np.expand_dims(X[:num_samples_train], axis=3)
    X_train_cos = np.expand_dims(Xcos[:num_samples_train], axis=3)
    y_train = y[:num_samples_train]
    X_test = np.expand_dims(X[num_samples_train:], axis=3)
    X_test_cos = np.expand_dims(Xcos[num_samples_train:], axis=3)
    y_test = y[num_samples_train:]
    X_train_cos.shape

    BATCH_SIZE = 64
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_cos, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test_cos, y_test))
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # Create the encoder and decoder models

    encoder, latent_space_dim = Encoder(attention=False)
    decoder = Decoder(latent_space_dim)
    zernike_decoder = latent_decoder(latent_space_dim)

    # Define the optimizer and the training params

    #EPOCHS = 100
    EPOCHS = 1
    validation_split = 0.1
    num_samples = len(train_dataset)
    num_train = round(num_samples * (1 - validation_split))
    train_ds = train_dataset.take(num_train).batch(BATCH_SIZE)
    val_dataset = train_dataset.skip(num_train).batch(BATCH_SIZE)
    autoencoder_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
    zernike_decoder_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
    record_ae = fit_ae(train_ds=train_ds, 
                    val_ds=val_dataset, 
                    test_ds=test_dataset, 
                    epochs=EPOCHS,
                    encoder=encoder,
                    decoder=decoder,
                    zernike_decoder=zernike_decoder,
                    cart=cart, #JV
                    autoencoder_optimizer=autoencoder_optimizer)
    
    
    print("JV4",cart)   
    EPOCHS = 150
    record_zae = fit_zae(train_ds=train_ds, 
                     test_ds=test_dataset, 
                     epochs=EPOCHS,
                     encoder=encoder,
                     decoder=decoder,
                     zernike_decoder=zernike_decoder,
                     zernike_decoder_optimizer=zernike_decoder_optimizer,
                     cart=cart) #JV
    print("Zernike total loss: ", record_zae['total_zernike_decoder_loss'][-1])
    print("Estimated Phase Loss: ", record_zae['estimated_phase_loss'][-1])
    print("Estimated Gradient Loss: ", record_zae['estimated_grad_loss'][-1])
    print("Zernike Decoder Loss: ", record_zae['zernike_decoder_loss'][-1])    

    # Save the models
    # encoder.save('./models/ae_encoder.h5')
    # decoder.save('./models/ae_decoder.h5')
    # zernike_decoder.save('./models/zae_zernike_decoder.h5')

    # Evaluate on test data
    np_config.enable_numpy_behavior()
    tf.config.run_functions_eagerly(True)
    zautoencoder = tf.keras.models.Model(
        inputs=encoder.inputs, 
        outputs=zernike_decoder(encoder.outputs)
    )
    zautoencoder.compile(optimizer=zernike_decoder_optimizer, 
                        loss=total_zernike_loss)
    zautoencoder.evaluate(X_test_cos, y_test)

    # Save the records
    with open('./records/ae_synthetic.pkl', 'wb') as f:
        pickle.dump(record_ae, f)
    with open('./records/zae_synthetic.pkl', 'wb') as f:
        pickle.dump(record_zae, f)

if __name__ == '__main__':
    main()
