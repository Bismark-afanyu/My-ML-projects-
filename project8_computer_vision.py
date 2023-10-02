import os
import cv2
import time
from tensorflow import keras
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from keras.callbacks import TensorBoard
#for image loading, resizing, and other image processing operations.
import numpy as np

Name = f'cat-vs-dog-prediction {int(time.time())}'
tensorboard = TensorBoard(log_dir= f'logs1\\{Name}\\')

# Set the path to the dataset directory
dataset_dir = "C:\\Users\\BANTA\\Desktop\\ML projects\\dataSets\\Cat_Dog_data"

# Seting the desired image size it determines the dimensions to which the images will be resized.
image_size = (100, 100)

# Function to preprocess the images 
# This function takes the path to the image directory as input.
def preprocess_images(image_dir):
    # Create an empty list for storing all the preprocessed images
    images = []
    labels = []

    # Iterate through the image directory
    # The glob module is used here to get a list of files in a given folder that match certain criteria.
    for category in os.listdir(image_dir):
        category_dir = os.path.join(image_dir, category)
        #Check if the category directory exists using os.path.isdir(category_dir). This ensures that only directories are considered, excluding any other files that might be present.        
        if os.path.isdir(category_dir):
            #Assign a numerical label to each category. cats are assigned the label 0 and dogs are assigned the label 1.
            label = 0 if category == "cats" else 1
            # Iterate through the images in the category directory
            for image_file in os.listdir(category_dir):
                #Create the image path by joining the category directory path with the image file name
                image_path = os.path.join(category_dir, image_file)

                # Load and resize the image
                image = cv2.imread(image_path)
                image = cv2.resize(image, image_size)

                # Normalize pixel values of the image by dividing it by 255.0. This scales the pixel values between 0 and 1.
                image = image.astype(np.float32) / 255.0

                # Append the image and label to the lists
                images.append(image)
                labels.append(label)

    # Convert the lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Preprocess the training and testing images
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")

train_images, train_labels = preprocess_images(train_dir)
test_images, test_labels = preprocess_images(test_dir)

#data visualization
# Count the number of samples in each class
cat_count = np.sum(train_labels == 0)
dog_count = np.sum(train_labels == 1)

# Create a bar plot to visualize the class distribution
plt.bar(['Cats', 'Dogs'], [cat_count, dog_count])
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()

# Display a few sample images from each class
cat_samples = train_images[train_labels == 0][:5]
dog_samples = train_images[train_labels == 1][:5]

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i in range(5):
    axes[0, i].imshow(cat_samples[i])
    axes[0, i].axis('off')
    axes[0, i].set_title('Cat')

    axes[1, i].imshow(dog_samples[i])
    axes[1, i].axis('off')
    axes[1, i].set_title('Dog')

plt.suptitle('Sample Images')
plt.show()


# Print the shape of the preprocessed data
print("Train images shape:", train_images.shape)
print("Train labels shape:", train_labels.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)

#create the model 
model = Sequential()
# Add Convolutional and MaxPooling layers
model.add(Conv2D(64, (3,3), activation = "relu"))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation = "relu"))
model.add(MaxPooling2D((2,2)))

# Flatten the output
model.add(Flatten())

# Add Dense layers
model.add(Dense(150, input_shape = train_images.shape[1:], activation= 'relu'))

model.add(Dense(2, activation= 'softmax'))

# Compile the model
model.compile(optimizer= "adam", loss= "sparse_categorical_crossentropy", metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=5, validation_split= 0.1, shuffle = True, batch_size= 32, callbacks= [tensorboard])
# Evaluate the model on the testing data
loss, accuracy = model.evaluate(test_images, test_labels)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Make predictions on the test data
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate precision, recall, and F1 score
precision = precision_score(test_labels, predicted_labels)
recall = recall_score(test_labels, predicted_labels)
f1 = f1_score(test_labels, predicted_labels)
# Print the training loss and test loss
train_loss = history.history['loss']
test_loss = history.history['val_loss']

print("Training Loss:", train_loss)
print("Test Loss:", test_loss)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# for i in range(len(predictions)):
#     if predictions[i][0] > predictions[i][1]:
#         prediction = [1., 0.]
#         if predictions[i][0] < predictions[i][1]:
#             prediction = [0., 1.]

#print("Model Accuracy:", predictions)


#Fine-tune the model
learning_rate = 0.0001
batch_size = 32
epochs = 10

# Unfreeze the last few layers for fine-tuning
for layer in model.layers[-5:]:
    layer.trainable = True

# Recompile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with fine-tuning
model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_images, test_labels))

# Evaluate the fine-tuned model on the testing data
loss, accuracy = model.evaluate(test_images, test_labels)
print("Test Loss (Fine-tuned):", loss)
print("Test Accuracy (Fine-tuned):", accuracy)


# Make predictions on new images
# Load and preprocess the new image
def predict_images(image_path):
    new_image = cv2.imread(image_path)
    new_image = cv2.resize(new_image, image_size)
    new_image = new_image.astype(np.float32) / 255.0
    new_image = np.expand_dims(new_image, axis=0)

    # Make predictions using the trained model
    predictions = model.predict(new_image)
    predicted_class = np.argmax(predictions)

    # Print the predicted class
    if predicted_class == 1:
        return("The Image passed is a Dog")
    else:
        return("The image passed is a Cat")


image_path = "C:\\Users\\BANTA\\Pictures\\Camera Roll\\dataset\\single_prediction\\cat2.jpg"
print(predict_images(image_path))
image_path = "C:\\Users\\BANTA\\Pictures\\Camera Roll\\dataset\\single_prediction\\cat_or_dog_2.jpg"
print(predict_images(image_path))
image_path = "C:\\Users\\BANTA\\Pictures\\Camera Roll\\dataset\\single_prediction\\cat_or_dog_1.jpg"
print(predict_images(image_path))
image_path = "C:\\Users\\BANTA\\Pictures\\Camera Roll\\dataset\\single_prediction\\dog.4014.jpg"
print(predict_images(image_path))
image_path = "C:\\Users\\BANTA\\Pictures\\Camera Roll\\dataset\\single_prediction\\dog.1.jpg"
print(predict_images(image_path))

# # Make predictions on new images
# def predict_images(image_path):
#     # Load and preprocess the new image
#     new_image = cv2.imread(image_path)
#     new_image = cv2.resize(new_image, image_size)
#     new_image = new_image.astype(np.float32) / 255.0
#     new_image = np.array(new_image)
#     # Reduce the dimensionality of the NumPy array to 1 or 2 dimensions.
#     if new_image.ndim > 2:
#         new_image = new_image.reshape(-1)
#     # Expand the numpy array of the single image to an array of images with one image.
#     new_images = np.expand_dims(new_image, axis=0)

#     # Make predictions using the trained model
#     predictions = model.predict(new_images)

#     # Print the predicted class
#     if predictions[0] == 1:
#         return("The Image passed is a Dog")
#     else:
#         return("The image passed is a Cat")


# image_path = "C:\\Users\\BANTA\\Pictures\\Camera Roll\\dataset\\single_prediction\\cat2.jpg"
# print(predict_images(image_path))
# image_path = "C:\\Users\\BANTA\\Pictures\\Camera Roll\\dataset\\single_prediction\\cat_or_dog_2.jpg"
# print(predict_images(image_path))
# image_path = "C:\\Users\\BANTA\\Pictures\\Camera Roll\\dataset\\single_prediction\\dog.4014.jpg"
# print(predict_images(image_path))








# Reshape the images to a 1D array
# train_images_flat = train_images.reshape(train_images.shape[0], -1)
# test_images_flat = test_images.reshape(test_images.shape[0], -1)


# # Initialize and train the SVM classifier
# svm_classifier = SVC(kernel='linear')
# svm_classifier.fit(train_images_flat, train_labels)

# # Make predictions on the test set using SVM
# svm_predictions = svm_classifier.predict(test_images_flat)

# # Calculate accuracy of SVM predictions
# svm_accuracy = accuracy_score(test_labels, svm_predictions)
# print("SVM Accuracy:", svm_accuracy)


# Make predictions on new images
# def predict_images(image_path):
#     # Load and preprocess the new image
#     new_image = cv2.imread(image_path)
#     new_image = cv2.resize(new_image, image_size)
#     new_image = new_image.astype(np.float32) / 255.0
#     new_image = np.array(new_image)
#     # Reduce the dimensionality of the NumPy array to 1 or 2 dimensions.
#     if new_image.ndim > 2:
#         new_image = new_image.reshape(-1)
#     # Expand the numpy array of the single image to an array of images with one image.
#     new_images = np.expand_dims(new_image, axis=0)

#     # Make predictions using the trained model
#     predictions = svm_classifier.predict(new_images)

#     # Print the predicted class
#     if predictions[0] == 1:
#         return("The Image passed is a Dog")
#     else:
#         return("The image passed is a Cat")


# image_path = "C:\\Users\\BANTA\\Pictures\\Camera Roll\\dataset\\single_prediction\\cat2.jpg"
# print(predict_images(image_path))
# image_path = "C:\\Users\\BANTA\\Pictures\\Camera Roll\\dataset\\single_prediction\\cat_or_dog_2.jpg"
# print(predict_images(image_path))
# image_path = "C:\\Users\\BANTA\\Pictures\\Camera Roll\\dataset\\single_prediction\\cat_or_dog_1.jpg"
# print(predict_images(image_path))
#print("Predicted Class:", predictions[0])

#from tensorflow import keras

# # Load the pre-trained model
# model = keras.applications.VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

# # Freeze the pre-trained layers
# for layer in model.layers:
#     layer.trainable = False

# # Add a new output layer for binary classification (cats and dogs)
# num_classes = 2
# model.layers.pop()
# output_tensor = model.layers[-1].output
# model.layers[-1].outbound_nodes = []

# output_layer = keras.layers.Dense(num_classes, activation='softmax')
# model_output = output_layer(output_tensor)
# model = keras.Model(inputs=model.input, outputs=model_output)

# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Print the model summary
# print(model.summary())

# # Train the model
# batch_size = 32
# epochs = 10

# model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_images, test_labels))

# # Evaluate the model on the testing data
# loss, accuracy = model.evaluate(test_images, test_labels)
# print("Test Loss:", loss)
# print("Test Accuracy:", accuracy)



# Fine-tune the model
# learning_rate = 0.0001
# batch_size = 32
# epochs = 10

# # Unfreeze the last few layers for fine-tuning
# for layer in model.layers[-5:]:
#     layer.trainable = True

# # Recompile the model
# model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train the model with fine-tuning
# model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_images, test_labels))

# # Evaluate the fine-tuned model on the testing data
# loss, accuracy = model.evaluate(test_images, test_labels)
# print("Test Loss (Fine-tuned):", loss)
# print("Test Accuracy (Fine-tuned):", accuracy)




# from sklearn.ensemble import BaggingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import GaussianNB, MultinomialNB
# from sklearn.metrics import accuracy_score

# # Initialize the base classifiers
# logistic_regression = LogisticRegression()
# svm = SVC(kernel='linear')
# random_forest = RandomForestClassifier()
# naive_bayes = GaussianNB()
# multinomial_naive_bayes = MultinomialNB()

# # Initialize the bagging classifier
# bagging_classifier = BaggingClassifier(base_estimator=None, n_estimators=10)

# # Fit the bagging classifier with the base classifiers
# bagging_classifier.fit(train_images_flat, train_labels)

# # Make predictions on the test set using the bagging classifier
# bagging_predictions = bagging_classifier.predict(test_images_flat)

# # Calculate accuracy of bagging predictions
# bagging_accuracy = accuracy_score(test_labels, bagging_predictions)
# print("Bagging Accuracy:", bagging_accuracy)
