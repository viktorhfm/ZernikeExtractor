from utils import *
from loss_functions import *
from tensorflow.python.ops.numpy_ops import np_config


def model_compiler(path, optimizer, loss):
    model = tf.keras.models.load_model(path, compile=False)
    model.compile(optimizer=optimizer, 
                  loss=loss)
    return model

def load_model(): 
    autoencoder_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
    zernike_decoder_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
    
    encoder = model_compiler('./models/ae_encoder.h5',
                       autoencoder_optimizer,
                       ae_loss)

    zernike_decoder = model_compiler('./saved_models/zae_zernike_decoder.h5',
                                zernike_decoder_optimizer,
                                total_zernike_loss)
    
    np_config.enable_numpy_behavior()
    tf.config.run_functions_eagerly(True)
    zautoencoder = tf.keras.models.Model(
        inputs=encoder.inputs, 
        outputs=zernike_decoder(encoder.outputs)
    )
    zautoencoder.compile(optimizer=zernike_decoder_optimizer, 
                        loss=total_zernike_loss)
    
    return zautoencoder

def inference():
    zautoencoder = load_model()
    zautoencoder.summary()
    print('Model loaded successfully')
    path = 'path.npy'
    image = np.expand_dims(np.load(path), axis=0)
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=3)
    zernike_coeff = zautoencoder.predict(image)
    print(f"The Zernike coefficients are: {zernike_coeff}")

if __name__ == '__main__':
    inference()
