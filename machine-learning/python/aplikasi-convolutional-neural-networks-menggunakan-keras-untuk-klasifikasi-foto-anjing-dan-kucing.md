---
description: >-
  Hasil belajar dari free course Convolutional Neural Networks from Scratch oleh
  Analytics Vidhya.
---

# Aplikasi Convolutional Neural Networks Menggunakan Keras untuk Klasifikasi Foto Anjing dan Kucing

{% hint style="info" %}
Kode di bawah ditulis dan dijalankan dengan menggunakan Visual Studio Code.
{% endhint %}

```python
# Mengimpor packages
import os
import numpy as np
import pandas as pd
import scipy
import sklearn
import keras
import scipy.misc
import imageio.v2 as imageio
from keras.models import Sequential
import cv2
from skimage import io
%matplotlib inline
```

```python
# Mendefinisikan letak fail
cat = os.listdir("/JupyterProject/CNN/cats_dogs/train/cats/")
dog = os.listdir("/JupyterProject/CNN/cats_dogs/train/dogs/")
filepath = "/JupyterProject/CNN/cats_dogs/train/cats/"
filepath2 = "/JupyterProject/CNN/cats_dogs/train/dogs/"
```

```python
# Memuat gambar
images = []
label = []
for i in cat:
    image = imageio.imread(filepath+i)
    images.append(image)
    label.append(0) # untuk gambar kucing
    
for i in dog:
    image = imageio.imread(filepath2+i)
    images.append(image)
    label.append(1) # untuk gambar anjing
```

```python
# Mengubah ukuran semua gambar
for i in range(0, 25000):
    images[i] = cv2.resize(images[i], (300, 300))
```

```python
# Mengubah gambar menjadi arrays
images = np.array(images)
label = np.array(label)
```

```python
# Mendefinisikan hyperparameter
filters = 10
filtersize = (5, 5)

epochs = 5
batchsize = 128

input_shape = (300, 300, 3)
```

```python
# Mengubah target variabel menjadi ukuran yang dibutuhkan
from keras.utils.np_utils import to_categorical
label = to_categorical(label)
```

```python
# Mendefinisikan model
model = Sequential()

model.add(keras.layers.InputLayer(input_shape=input_shape))

model.add(keras.layers.convolutional.Conv2D(filters, filtersize, strides=(1, 1),
padding='valid', data_format="channels_last", activation='relu'))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(units=2, input_dim=50, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(images, label, epochs=epochs, batch_size=batchsize, validation_split=0.3)

model.summary
```

```
Epoch 1/5
137/137 [==============================] - 488s 4s/step - loss: 36.6430 - accuracy: 0.5908 - val_loss: 2.1438 - val_accuracy: 0.3472
Epoch 2/5
137/137 [==============================] - 570s 4s/step - loss: 0.7166 - accuracy: 0.7257 - val_loss: 2.3494 - val_accuracy: 0.2455
Epoch 3/5
137/137 [==============================] - 315s 2s/step - loss: 0.4551 - accuracy: 0.8146 - val_loss: 2.6909 - val_accuracy: 0.2268
Epoch 4/5
137/137 [==============================] - 330s 2s/step - loss: 0.3620 - accuracy: 0.8679 - val_loss: 2.2331 - val_accuracy: 0.2699
Epoch 5/5
137/137 [==============================] - 330s 2s/step - loss: 0.3087 - accuracy: 0.8907 - val_loss: 2.9072 - val_accuracy: 0.2256





<bound method Model.summary of <keras.engine.sequential.Sequential object at 0x000001C07AB22B60>>
```
