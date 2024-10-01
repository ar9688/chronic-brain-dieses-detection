import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Part 1 - Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary',
                                                 shuffle=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary',
                                            shuffle=False)

# Part 2 - Extracting features and labels
# Extracting data from the generator
def extract_features(generator):
    X = []
    y = []
    for i in range(len(generator)):
        X_batch, y_batch = generator[i]
        X.append(X_batch)
        y.append(y_batch)
    return np.vstack(X), np.hstack(y)

# Extract training and test set features and labels
X_train, y_train = extract_features(training_set)
X_test, y_test = extract_features(test_set)

# Flatten the image data for SVM input (from 64x64x3 to 64*64*3)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Standardize the data
scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)
X_test_flat = scaler.transform(X_test_flat)

# Part 3 - SVM Model Definition
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)

# Part 4 - Training the SVM Model
svm_classifier.fit(X_train_flat, y_train)

# Part 5 - Evaluating the SVM Model
y_pred = svm_classifier.predict(X_test_flat)

# Calculating classification report including F1-score
report = classification_report(y_test, y_pred, target_names=['no', 'yes'])
print(report)

# Part 6 - Plotting confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['no', 'yes'])
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Part 7 - Making a Single Prediction
# Load and preprocess the image for single prediction
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/yes_or_no.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = scaler.transform(test_image.reshape(1, -1))  # Flatten and scale the image

# Predict
result = svm_classifier.predict(test_image)
prediction = 'yes' if result[0] == 1 else 'no'
print(prediction)
