import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras

import constraints
from repository import get_dataset
import os


def draw_curves(history, key1='accuracy', ylim1=(0.8, 1.1),
                key2='loss', ylim2=(0.0, 1.1)):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history[key1], "r--")
    plt.plot(history.history['val_' + key1], "g--")
    plt.ylabel(key1)
    plt.xlabel('Epoch')
    plt.ylim(ylim1)
    plt.legend(['train', 'test'], loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(history.history[key2], "r--")
    plt.plot(history.history['val_' + key2], "g--")
    plt.ylabel(key2)
    plt.xlabel('Epoch')
    plt.ylim(ylim2)
    plt.legend(['train', 'test'], loc='best')

    plt.show()


def plot_value_img(i, predictions, true_label, img,train_labels):
    predictions, true_label, img = predictions[i], true_label[i], img[i]
    predicted_label = np.argmax(predictions)
    true_value = np.argmax(true_label)

    plt.figure(figsize=(12, 10))

    path=constraints.DATA_PATH+'/Test'
    files = os.listdir(path)
    tablica = [i for i in range(len(predictions))]

    thisplot = plt.barh(tablica, predictions, color="gray")
    thisplot[predicted_label].set_color('r')
    thisplot[true_value].set_color('g')


    if predicted_label == true_value:
        color = 'green'
    else:
        color = 'red'
    plt.yticks(tablica,files)
    plt.show()

def run(data_path: str, epochs: int, size: tuple):
    train_images, train_labels, test_images, test_labels = get_dataset(data_path, size)

    classes = len(next(os.walk(data_path + "\\Test"))[1])

    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size[0], size[1], 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    lastmodel=model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))
    y_val_pred = model.predict(test_images)

    plot_value_img(1, y_val_pred,train_images, test_labels,test_images)
    draw_curves(lastmodel, key1='accuracy', ylim1=(0.7, 0.95), key2='loss', ylim2=(0.0, 0.8))
    loss, accuracy = model.evaluate(test_images, test_labels)
    return accuracy

