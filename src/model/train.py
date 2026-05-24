import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import cv2
import numpy as np


IMG_SIZE = (224, 224) 
BATCH_SIZE = 32 # num of images at a time during training

def custom_augmentation(img):
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Random Gaussian blur
    if np.random.rand() < 0.5:
        img = cv2.GaussianBlur(img, (3,3), 0)

    return img.astype(np.float32)

# Data augmentations
train_datagen = ImageDataGenerator(
    preprocessing_function=custom_augmentation,
    rescale=1./255,                     # Normalize pixel values
    validation_split=0.2,               # 20 percent for validation
    horizontal_flip=True,               # Randomly flip images (horizontally)
    rotation_range=10,                  # Randomly rotate images
    brightness_range=[0.8, 1.2],         # Randlomly adjust brightness
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Validation (normalized lang dito)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Load images from the split data (training)
train_generator = train_datagen.flow_from_directory(
    "data/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',        # open: 0  close: 1
    subset='training'           # 80% used for training
)

# Load images from the split data (validation)
val_generator = val_datagen.flow_from_directory(
    "data/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary', 
    subset='validation'         # 20% used for validation
)

# Load the pre-trained model
base_model = MobileNetV2(
    input_shape=(224, 224, 3),      
    include_top=False,
    weights='imagenet'          # pre-trained weights ito
)

# Freeze base layers (to stop updating the pre-trained weights)
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

# Add custom layers
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs=base_model.input, outputs=output)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop]
)


os.makedirs("models", exist_ok=True)

# Save model
model.save("models/model.h5")

print("Model training complete and saved.")
