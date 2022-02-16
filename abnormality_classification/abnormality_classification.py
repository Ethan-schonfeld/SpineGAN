#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# In[ ]:


# Generate the Dataset


# In[ ]:


image_size = (128, 128)
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    "../training_data", labels='inferred', label_mode='int',
    class_names=["normal", "abnormal"], color_mode='grayscale', batch_size=batch_size, image_size=image_size, 
    shuffle=True, seed=55, validation_split=0.2, subset="training",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "../training_data", labels='inferred', label_mode='int',
    class_names=["normal", "abnormal"], color_mode='grayscale', batch_size=batch_size, image_size=image_size, 
    shuffle=True, seed=55, validation_split=0.2, subset="validation"
)


# In[ ]:


plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")


# In[ ]:


data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)


# In[ ]:


train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)


# In[ ]:


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


# In[ ]:


model = make_model(input_shape=image_size + (1,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)


# In[ ]:


model.summary()


# In[ ]:


epochs = 50

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)


# In[ ]:


from sklearn.metrics import roc_auc_score


# In[ ]:


model = keras.models.load_model('./save_at_11.h5')


# In[ ]:


train_labels = []
train_predictions = []
for train_image, train_label in train_ds:
    train_prediction = model.predict(train_image)
    for val in train_label:
        train_labels.append(val)
    for val in train_prediction:
        train_predictions.append(val)


# In[ ]:


val_labels = []
val_predictions = []
for val_image, val_label in val_ds:
    val_prediction = model.predict(val_image)
    for val in val_label:
        val_labels.append(val)
    for val in val_prediction:
        val_predictions.append(val)


# In[ ]:


print("train:")
roc_auc_score(train_labels, train_predictions)


# In[ ]:


print("val:")
roc_auc_score(val_labels, val_predictions)


# In[ ]:


from sklearn.metrics import f1_score


# In[ ]:


targets_val = []
for pred in val_predictions:
    if pred > 0.5:
        targets_val.append(1)
    else:
        targets_val.append(0)


# In[ ]:


f1_score(val_labels, targets_val)


# In[ ]:




