from tensorflow import keras
from repository import get_dataset
import os
import constraints as CONST


def run(data_path: str, epochs: int, target_img_size: tuple):
    train_images, train_labels, test_images, test_labels = get_dataset()
    print(len(next(os.walk(data_path + "\\Test"))[1]))
    classes = len(next(os.walk(data_path + "\\Test"))[1])

    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(CONST.TARGET_IMAGE_HEIGHT, CONST.TARGET_IMAGE_WIDTH, 3)),
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
    model.fit(train_images, train_labels, epochs=CONST.EPOCHS, validation_data=(test_images, test_labels))

    loss, accuracy = model.evaluate(test_images, test_labels)
    print("Accuracy:" + str(accuracy))


run()
