from data_generators import zernike2cos, zernike2phi, zernike2gradient
import tensorflow as tf

loss_object = tf.keras.losses.MeanSquaredError()
BETA = 100
LAMBDA = 10
ALPHA = 1

loss_object_zernike = tf.keras.losses.MeanAbsoluteError()
loss_object_phase = tf.keras.losses.MeanAbsoluteError()

def ae_loss(y_true, y_pred):
    return BETA * loss_object(y_true, y_pred)

def cos_loss(y_true, y_pred):
    phi = zernike2cos(y_true)
    hat_phi = zernike2cos(y_pred)
    return LAMBDA * loss_object_phase(phi, hat_phi)

def phase_loss(y_true, y_pred, cart): #JV
    print("JV2",cart)
    phi = zernike2phi(y_true.numpy(), cart) #JV
    hat_phi = zernike2phi(y_pred, cart) #JV
    return ALPHA * loss_object_phase(phi, hat_phi)

def grad_loss(y_true, y_pred):
    dx_true, dy_true = zernike2gradient(y_true)
    dx_pred, dy_pred = zernike2gradient(y_pred)
    return LAMBDA * (0.5*loss_object(dx_true, dx_pred) + 0.5*loss_object(dy_true, dy_pred))
    
def zernike_loss(y_true, y_pred):
    return ALPHA * loss_object_zernike(y_true, y_pred)

def total_zernike_loss(y_true, y_pred):
    return phase_loss(y_true, y_pred) + grad_loss(y_true, y_pred) + zernike_loss(y_true, y_pred) #+ cos_loss(y_true, y_pred)
