import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import TFSMLayer
from tensorflow import keras

IMAGE_SIZE = 224
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "C:\\Projects\\Nivedi-Poultry\\training\\Poultry_disease",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)


class_name= dataset.class_names
class_name

def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1,shuffle=True, shuffle_size=10000):
    ds_size=len(ds)
    if shuffle:
        ds=ds.shuffle(shuffle_size, seed=12)
    train_size=int(train_split*ds_size)
    val_size=int(val_split*ds_size)
    train_ds=ds.take(train_size)
    val_ds=ds.skip(train_size).take(val_size)
    test_ds=ds.skip(train_size).skip(val_size)
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds=get_dataset_partitions_tf(dataset)
model=keras.layers.TFSMLayer("C:\\Projects\\Nivedi-Poultry\\models\\1", call_endpoint='serving_default')
for images_batch, labels_batch in test_ds.take(1):
    # Display the first image
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()

    print("First image to predict")
    plt.imshow(first_image)
    plt.show()
    print("Real label:", class_name[first_label])

    # Predict using the TFSMLayer
    batch_prediction = model(images_batch)  # Directly call the TFSMLayer
    first_prediction = batch_prediction['output_0'][0]
    predicted_class_index = np.argmax(first_prediction)
    predicted_label = class_name[predicted_class_index]
    
    print("Predicted label:", predicted_label)

