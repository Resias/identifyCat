import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from PIL import Image
import os
import shutil
import cv2

# cat image
image_directory = './photos/cat and dog/training_set/training_set/cats'
image_size = (500,500)

images = []
labels = []

# 이미지 데이터 불러오기
for filename in os.listdir(image_directory):
    if filename.endswith('.jpg'):
        img = cv2.imread(os.path.join(image_directory, filename))
        img = cv2.resize(img, image_size)
        images.append(img)
        labels.append(1)



# dog image

image_directory = './photos/cat and dog/training_set/training_set/dogs'
image_size = (500,500)


# 이미지 데이터 불러오기
for filename in os.listdir(image_directory):
    if filename.endswith('.jpg'):
        img = cv2.imread(os.path.join(image_directory, filename))
        img = cv2.resize(img, image_size)
        images.append(img)
        labels.append(0)

#print(images[0][2])
#print(images[0].shape)

import matplotlib.pyplot as plt
#plt.imshow(images[0])
#plt.imshow(images[-3])


# train-test 데이터 분할
images = np.array(images)
images, labels = shuffle(images, labels)

scaler = MinMaxScaler()
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2)

#print(train_images.dtype)
#print(train_images.shape)

train_images[3] = train_images[3]/255.0
train_scaled = train_images
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_labels, test_size=0.2)

train_scaled = np.array(train_scaled)
train_target = np.array(train_target, dtype=np.float32)
val_scaled = np.array(val_scaled)
val_target = np.array(val_target, dtype=np.float32)

#model 구축
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32,kernel_size=2,padding='same',activation='relu', input_shape=(500,500,3)))
model.add(tf.keras.layers.MaxPool2D(2))
model.add(tf.keras.layers.Conv2D(16,kernel_size=2,padding='same',activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(4,activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

model.summary()

from tensorflow.python.framework.ops import executing_eagerly_outside_functions
model.compile(optimizer='adam',loss='binary_crossentropy',metrics='accuracy')
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('best-CatCNN-model.h5')
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)
history = model.fit(train_scaled, train_target, epochs=100,
                    validation_data=(val_scaled,val_target),
                    callbacks=[checkpoint_cb,early_stopping_cb])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show()

model = tf.keras.models.load_model('best-CatCNN-model.h5')
model.evaluate(val_scaled,val_target)
