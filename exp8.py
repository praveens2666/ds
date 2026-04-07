import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# -------------------------------
# 1. Dataset paths
# -------------------------------
train_dir = "archive\seg_train\seg_train"
test_dir = "archive\seg_test\seg_test"

# -------------------------------
# 2. Image preprocessing
# -------------------------------
train_gen = ImageDataGenerator(rescale=1.0 / 255)
test_gen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# -------------------------------
# 3. Build CNN model
# -------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')
])

# -------------------------------
# 4. Compile model
# -------------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# 5. Train model
# -------------------------------
model.fit(
    train_data,
    epochs=5,
    validation_data=test_data
)

# -------------------------------
# 6. Evaluate model
# -------------------------------
loss, acc = model.evaluate(test_data)
print("Accuracy:", acc)

# -------------------------------
# 7. User Input Prediction
# -------------------------------
img_path = input("Enter image path: ")

img = load_img(img_path, target_size=(128, 128))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

# Get class labels (FIXED mapping)
class_labels = dict((v, k) for k, v in train_data.class_indices.items())
predicted_class = class_labels[np.argmax(prediction)]

print("Prediction:", predicted_class)