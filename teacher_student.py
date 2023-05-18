from tensorflow import keras
from repository import get_dataset
import os
import tensorflow as tf


def run_teacher_student(data_path: str, epochs: int, size: tuple):
    train_images, train_labels, test_images, test_labels = get_dataset(data_path, size)
    classes = len(next(os.walk(data_path + "\\Test"))[1])
    teacher_model = keras.Sequential([
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
    teacher_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    teacher_model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images[:50], test_labels[:50]))

    student_model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(512, input_shape=(None, classes), activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(classes, activation='softmax')
    ])

    for layer in teacher_model.layers:
        layer.trainable = False

    student_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    student_model.fit(train_images, teacher_model.predict(train_images), epochs=epochs, validation_data=(test_images[:50], test_labels[:50]))

    teacher_model.summary()
    student_model.summary()

    loss, accuracy = student_model.evaluate(test_images, test_labels)
    return accuracy
