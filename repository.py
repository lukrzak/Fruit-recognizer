from tensorflow.keras.preprocessing.image import ImageDataGenerator

import constraints as CONST


def get_dataset(type: str):
    path = CONST.TRAIN_DATA_PATH if type == "train" else CONST.TEST_DATA_PATH

    data_generator = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    dataset_generator = data_generator.flow_from_directory(
        path,
        target_size=(CONST.TARGET_IMAGE_HEIGHT, CONST.TARGET_IMAGE_WIDTH),
        batch_size=CONST.BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )

    x_data, y_data = dataset_generator.next()
    return x_data, y_data
