from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

import constraints as CONST


def get_dataset():
    data_generator = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_dataset_generator = data_generator.flow_from_directory(
        CONST.TRAIN_DATA_PATH,
        target_size=(CONST.TARGET_IMAGE_HEIGHT, CONST.TARGET_IMAGE_WIDTH),
        class_mode='categorical',
        batch_size=CONST.BATCH_SIZE,
        shuffle=True
    )

    test_dataset_generator = data_generator.flow_from_directory(
        CONST.TEST_DATA_PATH,
        target_size=(CONST.TARGET_IMAGE_HEIGHT, CONST.TARGET_IMAGE_WIDTH),
        class_mode='categorical',
        batch_size=CONST.BATCH_SIZE,
        shuffle=True
    )

    print("Test data processing")
    test_images, test_labels = [], []
    for x_batch, y_batch in test_dataset_generator:
        test_images.append(x_batch)
        test_labels.append(y_batch)
        if len(test_images) * test_dataset_generator.batch_size >= len(test_dataset_generator.filenames):
            break
    test_images = np.concatenate(test_images)
    test_labels = np.concatenate(test_labels)

    print("Train data processing")
    train_images, train_labels = [], []
    for x_batch, y_batch in train_dataset_generator:
        train_images.append(x_batch)
        train_labels.append(y_batch)
        if len(train_images) * train_dataset_generator.batch_size >= len(train_dataset_generator.filenames):
            break
    train_images = np.concatenate(train_images)
    train_labels = np.concatenate(train_labels)

    print("Processing finished")

    return train_images, train_labels, test_images, test_labels
