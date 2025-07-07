import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

def discriminator_spectral(input_shape=(128, 512, 1), name="Discriminator_Spectral"):
    """
    A PatchGAN with Spectral Normalization stability.
    
    """
    inputs = layers.Input(shape=input_shape)
    
    
    # L1: No normalization
    x = layers.Conv2D(32, 4, strides=2, padding='same',
                      kernel_initializer=tf.random_normal_initializer(0., 0.02))(inputs)
    x = layers.LeakyReLU(0.2)(x)
    
    # L2: Conv2D layer with SpectralNormalization
    x = tfa.layers.SpectralNormalization(
        layers.Conv2D(64, 4, strides=2, padding='same',
                      kernel_initializer=tf.random_normal_initializer(0., 0.02))
    )(x)
    x = layers.LeakyReLU(0.2)(x)

    # L3: Conv2D layer with SpectralNormalization
    x = tfa.layers.SpectralNormalization(
        layers.Conv2D(128, 4, strides=2, padding='same',
                      kernel_initializer=tf.random_normal_initializer(0., 0.02))
    )(x)
    x = layers.LeakyReLU(0.2)(x)
  
    # L4: Conv2D layer with SpectralNormalization
    x = tfa.layers.SpectralNormalization(
        layers.Conv2D(256, 4, strides=1, padding='same',
                      kernel_initializer=tf.random_normal_initializer(0., 0.02)) #ridotto numero di filtri da 265
    )(x)
    x = layers.LeakyReLU(0.2)(x)

    # The final output layer
    last = tfa.layers.SpectralNormalization(
        layers.Conv2D(1, 4, strides=1, padding='same',
                      kernel_initializer=tf.random_normal_initializer(0., 0.02))
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=last, name=name)