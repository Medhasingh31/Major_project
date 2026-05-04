import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def conv_block(inputs: tf.Tensor, filters: int) -> tf.Tensor:
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x


def build_light_unet(image_size: int = 256, base_filters: int = 16) -> keras.Model:
    """
    Builds a lightweight U-Net architecture suitable for moderate systems.
    The U-Net shape downsamples the image to extract features, then upsamples 
    it back to the original size to generate a pixel-by-pixel mask.
    """
    inputs = keras.Input(shape=(image_size, image_size, 3))

    # Encoder (Downsampling path): Extract features and reduce spatial dimensions
    c1 = conv_block(inputs, base_filters)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, base_filters * 2)
    p2 = layers.MaxPooling2D()(c2)

    # Bottleneck: The deepest part of the network
    c3 = conv_block(p2, base_filters * 4)

    # Decoder (Upsampling path): Reconstruct spatial dimensions and combine with encoder features
    u2 = layers.UpSampling2D()(c3)
    u2 = layers.Concatenate()([u2, c2])  # Skip connection from c2
    c4 = conv_block(u2, base_filters * 2)

    u1 = layers.UpSampling2D()(c4)
    u1 = layers.Concatenate()([u1, c1])  # Skip connection from c1
    c5 = conv_block(u1, base_filters)

    # Final Output Layer: 1x1 Convolution with Sigmoid activation for binary classification (Road = 1, Background = 0)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c5)
    
    return keras.Model(inputs, outputs, name="light_road_unet")


def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return 1.0 - dice_coefficient(y_true, y_pred)


def compile_model(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    """
    Compiles the model with standard configurations for binary segmentation.
    - Optimizer: Adam (efficient for training)
    - Loss: Binary Crossentropy (ideal for pixel-wise 0 or 1 classification)
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[dice_coefficient, "binary_accuracy"],
    )
    return model
