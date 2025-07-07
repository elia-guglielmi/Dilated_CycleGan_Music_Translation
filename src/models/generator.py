import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa


def asymmetric_dilated_resnet_block(input_tensor, filters, dilation_rate=(1, 1)):
    """
    A dilated residual block that accepts separate dilation rates for height and width.
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    
    # Main path
    x = layers.Conv2D(filters, 3, padding='same', dilation_rate=dilation_rate,
                      kernel_initializer=initializer, use_bias=False)(input_tensor)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(filters, 3, padding='same', dilation_rate=dilation_rate,
                      kernel_initializer=initializer, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)

    # Squeeze-and-Excitation block
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(filters // 8, activation='relu', kernel_initializer='he_normal')(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal')(se)
    x = layers.Multiply()([x, se]) # Re-weigh the feature maps
    
    # Add the shortcut connection
    x = layers.Add()([input_tensor, x])
    return x

def generator_with_dilated_cnn(input_shape=(128, 512, 2), name="Generator"):
    """
    Builds a generator with a U-Net architecture.
    - dilated ResNet blocks in the bottleneck.
    """
    inputs = layers.Input(shape=input_shape)
    initializer = tf.random_normal_initializer(0., 0.02)

    # Encoder
    down_stack = [
        (64, 4, False),    # (filters, kernel_size, apply_norm)
        (128, 4, True),
        (256, 4, True),
        (512, 4, True),
    ]

    x = inputs
    skips = []
    for filters, size, apply_norm in down_stack:
        x = layers.Conv2D(filters, size, strides=2, padding='same',
                          kernel_initializer=initializer, use_bias=False)(x)
        if apply_norm:
            x = tfa.layers.InstanceNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        skips.append(x)

    # ---Bottleneck with Dilated ResNet Blocks ---
    
    # Using 8 blocks instead of the 6 standard from the original CycleGAN ResNet.
    num_bottleneck_blocks = 8 
    # This pattern only expands along the time (width) axis.
    time_dilation_pattern = [1, 2, 4, 8, 1, 2, 4, 8] # Hybrid Dilated Convolutions to avoid gridding
    
    for i in range(num_bottleneck_blocks):
        # Dilation rate is (height, width). We keep height dilation at 1.
        rate = (1, time_dilation_pattern[i % len(time_dilation_pattern)])
        x = asymmetric_dilated_resnet_block(x, filters=512, dilation_rate=rate)
        
    # --- Upsampling Path (Decoder) with Skip Connections ---
    skips = reversed(skips[:-1])
    up_stack = [
        (256, 4, True),   # (filters, kernel_size, apply_dropout)
        (128, 4, True),
        (64, 4, False),
    ]

    for (filters, size, apply_dropout), skip in zip(up_stack, skips):
        x = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        x = layers.Conv2D(filters * 2, size, strides=1, padding='same', 
                          kernel_initializer=initializer, use_bias=False)(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Concatenate()([x, skip])
        
        
        x = layers.Conv2D(filters, size, strides=1, padding='same',
                          kernel_initializer=initializer, use_bias=False)(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        if apply_dropout:
            # dropout  
            x = layers.Dropout(0.3)(x) 

    # Final Layer
    x = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
    x = layers.Conv2D(2, 4, strides=1, padding='same',
                     kernel_initializer=initializer,
                     activation='tanh')(x) # tanh activation scales output to [-1, 1]

    return tf.keras.Model(inputs=inputs, outputs=x, name=name)
