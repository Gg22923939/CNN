# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 17:12:55 2023
此程式基於此篇論文修改
https://onlinelibrary.wiley.com/doi/full/10.1002/lary.29302
@author: user
"""


#%% 匯入所需的模組

import tensorflow as tf
import pathlib

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras import optimizers
import os
import numpy as np

tf.test.is_gpu_available()

#%%


num_classes = 3

input_size = 299

# 定義輸入張量
input_tensor = Input(shape=(input_size, input_size, 3))
# 使用 Xception 預訓練模型，不包含頂層（全連接層）
base_model = Xception(input_tensor=input_tensor, weights='imagenet', include_top=False)

# 在基模型之上加上全局平均池化層
x = base_model.output
x = GlobalAveragePooling2D()(x)

# 加上一個全連接層
x = Dense(1024, activation='relu')(x)

# 最後加上分類層，使用 softmax 啟動函數
predictions = Dense(num_classes, activation='softmax')(x)

# 建立模型
model = Model(inputs=[input_tensor], outputs=predictions)

# 選擇優化器和設定模型的編譯
rms = optimizers.Adam(lr=0.001)
model.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy'])

# 設定儲存模型的目錄和檔案命名規則
save_dir = os.path.join(r'C:\Users\user\Desktop\Python_Script\CNN\saved_models_Xception')
model_name = "model_{epoch:03d}-{val_acc:.3f}.hdf5"

# 若目錄不存在，則建立目錄
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# 設定模型檢查點，用於保存最佳模型
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=False)

# 設定 TensorBoard 回呼，用於視覺化訓練過程
tb_log_dir = os.path.join(os.getcwd(),'tb_log')
if not os.path.exists(tb_log_dir):
    os.makedirs(tb_log_dir)
tb = TensorBoard(log_dir=tb_log_dir, write_images=True)

# 設定學習速率調整回呼。自適應調整策略；一旦驗證損失在五個時期內停止改善，學習率將降低 0.3 倍；
lr_reducer = ReduceLROnPlateau(factor=0.3,
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

# 將所有回呼組成列表
callbacks = [checkpoint, lr_reducer, tb]



#%%

# 設定訓練資料和驗證資料的影像增強生成器
train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

# 透過影像生成器從目錄中讀取訓練和驗證資料
train_generator = train_datagen.flow_from_directory(
        r'C:\Users\user\Desktop\Python_Script\CNN\cats_and_dogs_and_Eardrums',  # 訓練資料目錄
        target_size=(input_size, input_size),  
        batch_size=14,
        class_mode='categorical')  

validation_generator = test_datagen.flow_from_directory(
        r'C:\Users\user\Desktop\Python_Script\CNN\cats_and_dogs_and_Eardrums',  # 驗證資料目錄
        target_size=(input_size, input_size),
        batch_size=8,
        class_mode='categorical')
#%%
# 使用生成器進行模型訓練
model.fit_generator(
        train_generator,
        steps_per_epoch=20,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=20,
        verbose=1)
