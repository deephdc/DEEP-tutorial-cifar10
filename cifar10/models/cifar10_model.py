import os
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
from keras import regularizers
import tensorflow as tf
K.set_image_dim_ordering('th')


labels=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def train_nn(epochs, lrate, outputpath):


    K.clear_session()
    #Training parameters
    epochs = epochs
    lrate = lrate
    decay = lrate/epochs

    # load data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # normalize inputs from 0-255 to 0.0-1.0
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Transform to hot vectors

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    print("We are running on %i classes" % num_classes)

    # Create the model

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(3, 32, 32)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax',kernel_regularizer=regularizers.l2(0.001)))

    # Compile model
    #sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
    print(model.summary())


    #Take only a certain number of samples
    nsamples=12000
    X_train=X_train[:nsamples,:,:,:]
    y_train=y_train[:nsamples,:]
    #Test only on a fraction
    nsamplestest=2000
    #X_test=X_train[:nsamplestest,:,:,:]
    #y_test=y_test[:nsamplestest,:]
    # Fit the model
    model.fit(X_train, y_train, validation_split=0.1,shuffle=True, epochs=epochs, batch_size=64)

    #Evaluate the model on the test dataset
    scores = model.evaluate(X_test, y_test, verbose=10)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    #Save the model
    model.save(os.path.join(outputpath,"model.h5"))

def predict_nn(image,outputpath):
    K.clear_session()

    #We load the model we have just trained
    model=load_model(os.path.join(outputpath, "model.h5"))

    #Expand the dimensions of the image to match the network architecture
    if K.image_data_format() == 'channels_first':
        image = image.reshape((1, 3, 32, 32))
    else:
        image = image.reshape((1, 32, 32, 3))


    predicted_vector = model.predict(image)
    # return 5 best predictions
    idxs = numpy.argsort(predicted_vector[0])[::-1][:5] 
    probs_best = []
    print("Top 5 predictions:")
    for i in idxs:
        print("{0:10s} :  {1:5.4f}".format(labels[i], predicted_vector[0][i]))

    # Print the prediction
    label_best = labels[idxs[0]]
    message= "The image is a " + label_best
    if label_best=="airplane" or label_best=="automobile":
        message= "The image is an " + label_best

    print()
    print(message)

    return message
