import os
import pandas as pd
import numpy as np
import seaborn as sn; sn.set(font_scale=1.4)
from sklearn.metrics import confusion_matrix
import cv2
import keras 
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt   
from tqdm import tqdm  
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers
from keras.models import load_model


#%%
'特徵提取'
'VGG16抓圖片特徵 >> 自己的模型'
from keras.applications.vgg16 import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False, # 捨棄原輸出層
                  input_shape=(150, 150, 3)) # 我的圖格式

conv_base.summary() # 模型參數

#%%
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20
'輸入 + 清洗 + vgg預測'
def extract_features(directory, sample_count): #路徑 樣本數
    features = np.zeros(shape=(sample_count, 4, 4, 512)) #最後層
    labels = np.zeros(shape=(sample_count)) #樣本數

    #輸入+清洗
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),  # 我的圖格式
        batch_size=20,   
        class_mode='binary')     # 兩類(cat & dog)
                                 # 多類categorical
    #vgg16生成特徵
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch) # 透過“卷積基底”來淬取圖像特徵
        features[i * batch_size : (i + 1) * batch_size] = features_batch # 把特徴先存放起來
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch #把標籤先存放起來
        i += 1
        if i * batch_size >= sample_count:
            break
    
    print('extract_features complete!')
    return features, labels

train_features, train_labels = extract_features(r'C:\Users\user\Desktop\Python_Practice\CNN\training_set', 2000) 
validation_features, validation_labels = extract_features(r'C:\Users\user\Desktop\Python_Practice\CNN\test_set', 500)

#%%
# 最後層(樣本數,4,4,512) 開始接>>展開
train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (500, 4 * 4 * 512))

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid')) # 2元

model.compile(optimizer='adam',
              loss='binary_crossentropy',metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))
#%%
model.evaluate(validation_features,  validation_labels, verbose=2)

#%%
'預測'
predictions = model.predict(validation_features)     # Vector of probabilities
pred_labels = [np.argmax(i) for i in predictions] # We take the highest probability
#%%
'混淆矩陣'
CM = confusion_matrix(validation_labels, pred_labels)
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 
print(accuracy(CM))
#%%
'混淆矩陣視覺化，看錯誤'
ax = plt.axes()
sn.heatmap(CM, annot=True, 
           annot_kws={"size": 10}, 
           fmt='d',
           xticklabels=['cat','dog'], 
           yticklabels=['cat','dog'], ax = ax)
ax.set_title('Confusion matrix')
plt.xlabel('Pred', fontsize = 20)
plt.ylabel('True', fontsize = 20)
plt.show()


#%%
'視覺化'
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, label='Training acc')
plt.plot(epochs, val_acc, label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#%%
model.save("85%_model")
model = load_model('CNN_model')