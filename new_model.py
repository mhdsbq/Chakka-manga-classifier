import itertools
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D ,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib

# Defining Importent variables

nb_train_samples = 1000
nb_validation_samples = 100
epochs = 50
batch_size = 20

# Defining Image Path

train_path = 'data/train'           # Test images
valid_path = 'data/validation'      # Validation images - used for validation while training
test_path = 'data/test'             # Test data used to plot confusion matrix after training
                                    # Confusion matrix is a simple plot to check accuracy of model.
                                    # Test accuracy and validation accuracy is plotted here
                                    # With this we can determine overwriting

# Creating training data, validation data and test data
"""We are using a pre processing function which is inbuilt to keras vgg16 model. vgg16 is a pretrained model but we are not using this model
.We are only utilizing the pre processing function of vgg16 model, this function subtracts mean values of Red,Blue and Green from each image."""

# Sheer_range, zoom and horizontal flip are used for IMAGE AUGMENTATION, ie: to make multiple instances of single image for better training accuracy.
# and hence we don't have to augment validation and testing images.

# Test batch is not shuffled because it is easy to analyse test accuracy
# Shuffling is done to avoid recognising of  pattern
train_batch=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
                               shear_range=0.2, zoom_range=0.2, horizontal_flip=True
                               ).flow_from_directory(directory=train_path, target_size=(224,224), classes=['jackfruit', 'mango'],batch_size=batch_size)
valid_batch=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=valid_path, target_size=(224,224,), classes=['jackfruit', 'mango'],batch_size=batch_size)
test_batch=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=test_path, target_size=(224,224,), classes=['jackfruit', 'mango'],batch_size=batch_size,shuffle=False)

# Calling next batch, here batch size is 20, we plot this batch to look through pre processed images.
imgs, labels = next(train_batch)

# A TensorFlow function, we copied directly from the tensorflow website
# This function will plot images in the form of a grid of 2 row with 10 columns each row
def plotImages(image_arr):
    fig, axes = plt.subplots(2, 10, figsize=(20,20))       # row, column and figure size
    axes = axes.flatten()
    for img, ax in zip(image_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(imgs)    # PLOT IMAGES.
print(labels)       # PRINT LABELS


# Creating a sequential neural network (Called a 'model' or 'Neural net' also)
model = Sequential([
    Conv2D(units=32, kernel_size=(3,3), input_shape=(224,224,3), padding='same', activation='relu'),   # refer:https://keras.io/api/layers/convolution_layers/convolution2d/
                                                                                    # images will have zero padding, Input shape is important because it is the first layer.
    MaxPooling2D(pool_size=(2,2)),                                                  # Down sampling. refer:https://keras.io/api/layers/pooling_layers/max_pooling2d/

    Conv2D(units=32,kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(units=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),                           # Convert 2D array to 1D array
    Dense(units=64, activation='relu'),  # Dense deeply connected layer. Refer:https://www.tutorialspoint.com/keras/keras_dense_layer.htm
    Dropout(0.5),                        # Dropout layer is added to reduce or prevent over-fitting. Refer:https://keras.io/api/layers/regularization_layers/dropout/
    Dense(units=2, activation='softmax') # Final layer has 2 units since we have two classifiers. 'softmax' will give probability corresponding to both classes.
])

# Printing out the built model
model.summary()

# Compile neural network model
# Adem optimizer is better than rmsprop by some of the articles
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# We can use one dense layer with sigmoid Activation and loss equals 'binary cross entropy',
# but it is only used for binary classification, Also Sigmoid does not give probability for each class


# model.fit will perform training. (it returns history)
# 'steps per epoch' and 'validation steps' are important params
model.fit(
            x=train_batch,
            steps_per_epoch= nb_train_samples // batch_size,
            validation_data=valid_batch,
            validation_steps= nb_validation_samples // batch_size,
            epochs=epochs,
            verbose=1)


# MODEL ACCURACY
# we are plotting training accuracy and validation accuracy . To determine over-fitting
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'],loc='upper left')
plt.show()

# MODEL LOSS
# we are plotting training loss and validation loss
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'],loc='upper left')
plt.show()

# Saving the model with its full structure, This can be loaded with 'keras.models.load_model("model_name.h5")'
model.save('model_jypter.h5')

# import a test batch for making confusion matrix
test_imgs, test_labels = next(test_batch)
plotImages(test_imgs)
print(test_labels)

# Print labels (jackfruit, mango)= (0 , 1) ..of each images in test batch
test_batch.classes()

# Predict Test batch
# This test batch is already pre processed in the beginning
predictions = model.predict(x=test_batch, verbose=0)

# Non rounded prediction
print(predictions)

# Rounded prediction ie: 0 or 1
np.round(predictions)

# Building confusion matrix
cm = confusion_matrix(y_true=test_batch.classes, y_pred=np.argmax(predictions, axis=-1))


#  function for Plotting confusion_matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Check labels
test_batch.class_indices()

# PLOTTING THE CONFUSION MATRIX
cm_plot_labels = ['chakka', 'manga']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')


#================ PREDICTION PART ===================#

# Load image as pixel data
from keras.preprocessing.image import load_img
# load image from file
image = load_img('data/test/jackfruit/978.jpg', target_size=(224, 224))

# Convert to a numpy array
from keras.preprocessing.image import img_to_array
image = img_to_array(image)

# Reshape image before preprocessing,   image reshaped from(224, 224, 3) to (1, 224, 224, 3)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# Prepare Image in the required form of vgg16, we trained our model with image preprocessing of vgg16.
# Our model is not vgg16, We just used the image pre processing function from vgg 16 (subtract mean RGB)
from keras.applications.vgg16 import preprocess_input
image = preprocess_input(image)

# Import the saved model
model_1 = keras.models.load_model('model_jypter.h5')
rslt = model_1.predict(image)

# Print Prediction Result
print(rslt)

# rounded result
round_rslt = np.round(rslt)
print(round_rslt)

# Final output
if str(round_rslt[0][0])[0]=='1':  # checking the value of jackfruit label.
    prediction = 'Chakka'
else:
    prediction = 'Manga'
print(prediction)
