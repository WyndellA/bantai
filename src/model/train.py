import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_SIZE = (224, 224) 
BATCH_SIZE = 32 # num of images at a time during training

# Data augmentations
train_datagen = ImageDataGenerator(
    rescale=1./255,                     # Normalize pixel values
    validation_split=0.2,               # 20 percent for validation
    horizontal_flip=True,               # Randomly flip images (horizontally)
    rotation_range=10,                  # Randomly rotate images
    brightness_range=[0.8, 1.2]         # Randlomly adjust brightness
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
base_model.trainable = False

# Add custom layers
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)


os.makedirs("models", exist_ok=True)

# Save model
model.save("models/model.h5")

print("Model training complete and saved.")