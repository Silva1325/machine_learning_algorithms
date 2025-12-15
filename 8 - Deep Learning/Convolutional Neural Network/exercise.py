import tensorflow as tf
import keras 

IMAGES_PATH = '8 - Deep Learning\Datasources\Domestic Animals'
BATCH_SIZE = 32
IMAGE_SIZE = 64

# Training dataset (without augmentation initially)
train_ds = keras.utils.image_dataset_from_directory(
    directory= r'8 - Deep Learning\Datasources\Domestic Animals\training_set',
    labels='inferred',
    label_mode='binary',
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SIZE, IMAGE_SIZE))

# Validation dataset
validation_ds = keras.utils.image_dataset_from_directory(
    directory= r'8 - Deep Learning\Datasources\Domestic Animals\test_set',
    labels='inferred',
    label_mode='binary',
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SIZE, IMAGE_SIZE))

# Define augmentation layers
train_datagen = keras.Sequential([
    keras.layers.Rescaling(1./255),
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomZoom(0.2),
    keras.layers.RandomContrast(0.2),
])

# Define test preprocessing (only rescaling, no augmentation)
test_datagen = keras.Sequential([
    keras.layers.Rescaling(1./255),
])

# Apply augmentation to training data
train_ds = train_ds.map(lambda x, y: (train_datagen(x, training=True), y))

# Apply preprocessing to validation data
validation_ds = validation_ds.map(lambda x, y: (test_datagen(x, training=False), y))

# Optional: Improve performance with prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)

# Build the CNN model
cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the CNN (now with augmentation!)
cnn.fit(x=train_ds, validation_data=validation_ds, epochs=25)

# Making a single prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img(r'8 - Deep Learning\Datasources\Domestic Animals\single_prediction\cat_or_dog_3.jpg', target_size=(IMAGE_SIZE, IMAGE_SIZE))
test_image = image.img_to_array(test_image)
test_image = test_image / 255.0  # Don't forget to rescale!
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)

if result[0][0] >= 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)