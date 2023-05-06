from tensorflow import keras
from repository import get_dataset
import os


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
    model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))

    loss, accuracy = model.evaluate(test_images, test_labels)
    return accuracy

