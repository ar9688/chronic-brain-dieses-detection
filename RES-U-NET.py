import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Input, Add, Dropout, Flatten, Dense
from keras.models import Model

# Part 1 - Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')


# Part 2 - ResUNet Architecture

# Residual Block definition
def residual_block(x, filters):
    shortcut = x
    # Apply 1x1 convolution to match the dimensions of the output
    shortcut = Conv2D(filters, (1, 1), padding='same')(shortcut)

    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters, (3, 3), activation=None, padding='same')(x)

    # Now both 'x' and 'shortcut' have the same shape
    x = Add()([x, shortcut])  # Add the shortcut to the main path
    x = tf.keras.layers.Activation('relu')(x)
    return x


# Encoder Block definition
def encoder_block(x, filters):
    x = residual_block(x, filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p


# Decoder Block definition
def decoder_block(x, skip, filters):
    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, skip])
    x = residual_block(x, filters)
    return x


# Building the ResUNet model
def build_resunet(input_shape):
    inputs = Input(input_shape)

    # Encoder
    skip1, pool1 = encoder_block(inputs, 32)
    skip2, pool2 = encoder_block(pool1, 64)
    skip3, pool3 = encoder_block(pool2, 128)
    skip4, pool4 = encoder_block(pool3, 256)
    skip5, pool5 = encoder_block(pool4, 512)

    # Bottleneck
    bottleneck = residual_block(pool5, 1024)

    # Decoder
    d1 = decoder_block(bottleneck, skip5, 512)
    d2 = decoder_block(d1, skip4, 256)
    d3 = decoder_block(d2, skip3, 128)
    d4 = decoder_block(d3, skip2, 64)
    d5 = decoder_block(d4, skip1, 32)

    # Classification head
    flatten = Flatten()(d5)
    dense1 = Dense(256, activation='relu')(flatten)
    dropout = Dropout(0.5)(dense1)
    output = Dense(1, activation='sigmoid')(dropout)

    # Model definition
    model = Model(inputs, output)
    return model


# Input shape
input_shape = (64, 64, 3)

# Build and compile the ResUNet model
resunet = build_resunet(input_shape)
resunet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 3 - Training the ResUNet Model
# Callbacks: Early Stopping and ReduceLROnPlateau
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

# Train the ResUNet model
history = resunet.fit(x=training_set, validation_data=test_set, epochs=50, callbacks=[early_stopping, lr_scheduler])

# Extracting accuracy and error rate from the training process
train_accuracy = history.history['accuracy'][-1]
test_accuracy = history.history['val_accuracy'][-1]
error_rate = 1 - test_accuracy

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Error Rate: {error_rate:.4f}")

# Part 4 - Plotting Accuracy and Loss
# Plotting training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Part 5 - Evaluating the Model and Calculating F1 Score
# Predicting the Test set results
y_pred = resunet.predict(test_set)
y_pred = np.where(y_pred > 0.5, 1, 0)

# Getting the true labels
y_true = test_set.classes

# Calculating the classification report including the F1 score
report = classification_report(y_true, y_pred, target_names=['no', 'yes'])
print(report)

# Part 6 - Making a Single Prediction
# Making a single prediction
test_image = image.load_img('dataset/single_prediction/yes_or_no.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = resunet.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'yes'
else:
    prediction = 'no'
print(prediction)
