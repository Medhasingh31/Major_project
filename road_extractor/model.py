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

    # Final Output Layer: 1x1 Conv with Sigmoid for binary road/background segmentation
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c5)

    return keras.Model(inputs, outputs, name="light_road_unet")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """
    Sørensen-Dice coefficient computed per-batch over all pixels.

    Flattening across (batch, H, W, 1) keeps the metric numerically stable
    for highly imbalanced road/background distributions (~5% road pixels).
    smooth=1.0 prevents division-by-zero on all-background batches.
    """
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def iou_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """
    Intersection-over-Union (Jaccard index) computed per-batch over all pixels.

    IoU = |A ∩ B| / |A ∪ B|  = intersection / (sum_true + sum_pred - intersection)

    Threshold: predictions are treated as soft probabilities; the product
    y_true * y_pred approximates the soft intersection, matching how Dice
    is computed so both metrics are consistent.
    """
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Dice loss = 1 − Dice coefficient.

    Directly optimises overlap between prediction and ground truth.
    Works well for class-imbalanced segmentation (road ≈ 5% of pixels).
    """
    return 1.0 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
     """
     Weighted Dice + Binary Cross-Entropy loss.

     Gives more importance to Dice overlap, which is useful for
     highly imbalanced road/background segmentation.
     """
     bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
     bce = tf.reduce_mean(bce)

     dice = dice_loss(y_true, y_pred)

     return 2.0 * dice + bce

# ---------------------------------------------------------------------------
# Model compilation
# ---------------------------------------------------------------------------

# Map CLI-friendly loss names to callables
LOSS_REGISTRY: dict[str, object] = {
    "combined": combined_loss,
    "dice":     dice_loss,
    "bce":      tf.keras.losses.BinaryCrossentropy(),
}

LOSS_DISPLAY_NAMES: dict[str, str] = {
    "combined": "Dice + Binary Cross-Entropy (combined)",
    "dice":     "Dice loss",
    "bce":      "Binary Cross-Entropy",
}


def compile_model(
    model: keras.Model,
    learning_rate: float = 1e-3,
    loss_name: str = "combined",
) -> keras.Model:
    """
    Compile the U-Net for binary road segmentation.

    Loss choices (pass via loss_name):
      'combined'  — Dice + BCE  [default, recommended for imbalanced data]
      'dice'      — Dice loss only
      'bce'       — Binary Cross-Entropy only

    Metrics tracked every epoch:
      - dice_coefficient  (primary validation metric for checkpointing)
      - iou_coefficient   (Jaccard index, standard for segmentation benchmarks)
      - binary_accuracy   (pixel-level accuracy, included for completeness)
    """
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{loss_name}'. Choose from: {list(LOSS_REGISTRY.keys())}"
        )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=LOSS_REGISTRY[loss_name],
        metrics=[dice_coefficient, iou_coefficient, "binary_accuracy"],
    )
    return model
