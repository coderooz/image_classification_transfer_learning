import tensorflow as tf

def prepare_data(image_size=(150, 150), batch_size=32):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest'
    )
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'data/train/',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    validation_generator = test_datagen.flow_from_directory(
        'data/validation/',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    return train_generator, validation_generator
