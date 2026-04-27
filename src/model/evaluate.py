import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load trained model
model = tf.keras.models.load_model("models/model.h5")

# Load test data (NO augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    "data/test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Predict
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int)

# True labels
true_classes = test_generator.classes

# Print results
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes))

print("\nConfusion Matrix:")
print(confusion_matrix(true_classes, predicted_classes))

print(test_generator.class_indices)