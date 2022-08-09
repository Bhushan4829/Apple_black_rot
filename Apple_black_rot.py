#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator as Imgen
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,Dropout,Flatten,Dense,GlobalAveragePooling2D,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.densenet import DenseNet201
import requests 

from sklearn.metrics import confusion_matrix,classification_report


# In[29]:


traingen = Imgen(preprocessing_function=preprocess_input,
                
                shear_range = 0.2,
                zoom_range = 0.2,
                width_shift_range = 0.2,
                height_shift_range = 0.2,
                fill_mode="nearest",
                validation_split=0.30)


testgen = Imgen(preprocessing_function=preprocess_input)


# In[30]:


trainds = traingen.flow_from_directory("D:\Apple\Apple_black_rot\Train",
                                      target_size=(150,150),
                                       class_mode="categorical",
                                       seed=123,
                                       batch_size=32,
                                       subset="training"
                                      )
valds = traingen.flow_from_directory("D:\Apple\Apple_black_rot\Train",
                                      target_size=(150,150),
                                       class_mode="categorical",
                                       seed=123,
                                       batch_size=32,
                                   subset="validation"
                                      )
testds = testgen.flow_from_directory("D:\Apple\Apple_black_rot\Valid",
                                    target_size=(150,150),
                                    class_mode="categorical",
                                    seed=123,
                                    batch_size=32,
                                    shuffle=False)


# In[31]:


c = trainds.class_indices
classes = list(c.keys())
classes


# In[ ]:





# In[32]:


x,y = next(trainds)
def plotImages(x,y):
    plt.figure(figsize=[15,11])
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(x[i])
        plt.title(classes[np.argmax(y[i])])
        plt.axis("off")
    plt.show()


# In[33]:


plotImages(x,y)


# In[34]:


resp = requests.get('http://www.google.com')
base_model = DenseNet201(include_top=False,
                     input_shape=(150,150,3),
                      weights = "imagenet",
                      pooling="avg"
                     )
base_model.trainable = False


# In[35]:


image_input = Input(shape=(150,150,3))

x = base_model(image_input,training = False)

x = Dense(512,activation = "relu")(x)

x = Dropout(0.3)(x)

x = Dense(128,activation = "relu")(x)

image_output = Dense(2,activation="softmax")(x)

model = Model(image_input,image_output)


# In[36]:


model.summary()


# In[37]:


model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


# In[38]:


my_calls = [EarlyStopping(monitor="val_accuracy",patience=3),
            ModelCheckpoint("Model.h5",verbose= 1 ,save_best_only=True)]


# In[39]:


hist = model.fit(trainds,epochs=4,validation_data=valds,callbacks=my_calls)


# In[40]:


model.evaluate(testds)


# In[41]:


plt.figure(figsize=(15,6))

plt.subplot(1,2,1)
plt.plot(hist.epoch,hist.history['accuracy'],label = 'Training')
plt.plot(hist.epoch,hist.history['val_accuracy'],label = 'validation')

plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(hist.epoch,hist.history['loss'],label = 'Training')
plt.plot(hist.epoch,hist.history['val_loss'],label = 'validation')

plt.title("Loss")
plt.legend()
plt.show()


# In[42]:


pred = model.predict(testds)


# In[43]:


pred = [np.argmax(i) for i in pred]


# In[44]:


y_test = testds.classes


# In[45]:


print(classification_report(pred,y_test))


# In[46]:


print(confusion_matrix(pred,y_test))


# In[47]:


plt.figure(figsize=(15,6))

plt.subplot(1,2,1)
plt.bar(hist.epoch,hist.history['accuracy'],label = 'Training')
plt.bar(hist.epoch,hist.history['val_accuracy'],label = 'validation')

plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.bar(hist.epoch,hist.history['loss'],label = 'Training')
plt.bar(hist.epoch,hist.history['val_loss'],label = 'validation')

plt.title("Loss")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




