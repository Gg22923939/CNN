## Codes are edited base on the following example: https://www.tensorflow.org/tutorials/images/transfer_learning

#%% Import module
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pathlib
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold

tf.test.is_gpu_available()

#%%
data_dir = os.path.join(r'C:\Users\user\Desktop\Python_Script\CNN\Eardrum_Four')
data_dir = pathlib.Path(data_dir).with_suffix('')
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


batch_size = 32
IMG_SIZE = (160, 160)

train_dataset = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size= IMG_SIZE,
  batch_size=batch_size)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=IMG_SIZE,
  batch_size=batch_size)

class_names = train_dataset.class_names
print(class_names)

# 查看每 batch 的圖片大小與標籤
for image_batch, labels_batch in train_dataset:
  print(image_batch.shape)
  print(labels_batch.shape)
  break



#%% Show the first nine images and labels form the training set:
    
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    
#%% Use buffered prefetching to load images from disk without having I/O become blocking.

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

#%% Using random data augmentation

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
    ]
)

for images, labels in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = images[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(
            tf.expand_dims(first_image, 0), training=True
        )
        plt.imshow(augmented_image[0].numpy().astype("int32"))
        plt.title(int(labels[0]))
        plt.axis("off")


#%% Create the base model from the pre-trained convnets

# Create the base model from the pre-trained model
IMG_SHAPE = IMG_SIZE + (3,)
print(IMG_SHAPE)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False, # By specifying the include_top=False argument, you load a network that doesn't include the classification layers at the top, which is ideal for feature extraction.
                                               weights='imagenet')

# This feature extractor converts each 160x160x3 image into a 5x5x1280 block of features
# Let's see what it does to an example batch of images:

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)              
print(feature_batch.shape)

#%% Feature Extraciton

# Freeze the convolutional base
base_model.trainable = False

# Let's take a look at the base model architecture
base_model.summary()


#%% Add a classification head

# =============================================================================
# # Method 1 設定一個 rescale 並且 map 到每個訓練集的圖片中
# rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)
# 
# train_dataset = train_dataset.map(lambda x, y: (rescale(x), y))
# image_batch, labels_batch = next(iter(train_dataset))
# first_image = image_batch[0]
# # Notice the pixel values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image))
# =============================================================================

# Method 2: Rescale pixel values 設定一個 rescale，並在建立模型的時候放進去
# The pixel values in your images are in [0, 255]. 我們要將像素轉換成 [0, 1] 或是 [-1, 1]（根據模型）

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)



#To generate predictions from the block of features, average over the spatial 5x5 spatial locations, using a tf.keras.layers.GlobalAveragePooling2D layer to convert the features to a single 1280-element vector per image.
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

# 最後加上分類層
# Apply a tf.keras.layers.Dense layer to convert these features into a single prediction per image. 
# If there is binary classification, You don't need an activation function because this prediction will be treated as a logit,
# or a raw prediction value. Positive numbers predict class 1, negative numbers predict class 0.

num_classes = len(class_names)
# 二分類以上的輸出層
prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)


inputs = tf.keras.Input(shape=(160, 160, 3))
# x = data_augmentation(inputs) # Apply random data augmentation
# x = preprocess_input(inputs)
x = rescale(inputs)
x = base_model(x, training=False)
# The training=False argument indicates that the model is in inference mode (not training)
x = global_average_layer(x) # 在 base_model 之上加上全局平均池化層
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

model.summary()

# 畫出模型的架構
tf.keras.utils.plot_model(model, show_shapes=True)


#%% Compile the model
# Compile the model before training it. Since there are two classes,
# use the tf.keras.losses.BinaryCrossentropy loss with from_logits=True since the model provides a linear output.


base_learning_rate = 1e-4
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#%% 設定 callbacks

# 設定檢查點的儲存路徑和檔名格式
checkpoint_filepath = r'C:\Users\user\Desktop\Python_Script\CNN\Models'
model_name = "model_{epoch:03d}-{val_accuracy:.3f}.h5"
# 若目錄不存在，則建立目錄
if not os.path.isdir(checkpoint_filepath):
    os.makedirs(checkpoint_filepath)
checkpoint_filepath = os.path.join(checkpoint_filepath, model_name)

# 設定 ModelCheckpoint 回呼
model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,  # 只儲存模型權重
    monitor='val_accuracy',  # 監控驗證集的準確度
    mode='max',  # 目標是最大化驗證集的準確度
    save_best_only=True,  # 只保存在驗證集上性能更好的模型
    verbose=1, # 顯示 log 資訊
)

# 設定學習速率調整回呼。自適應調整策略；一旦驗證損失在五個時期內停止改善，學習率將降低 0.3 倍；
lr_reducer = ReduceLROnPlateau(factor=0.3,
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
# 將所有回呼組成列表
callbacks = [model_checkpoint, lr_reducer]

#%% Train the model
initial_epochs = 10

loss0, accuracy0 = model.evaluate(validation_dataset)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset,
                    callbacks=callbacks)

# =============================================================================
# 將儲存的檢查點調用
# model.load_weights(r'C:\Users\user\Desktop\Python_Script\CNN\Models\model_007-0.823.h5') #檔案名稱修改成想要的
# =============================================================================
#%% Learning curves
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.xlim([0, initial_epochs])
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.xlim([0, initial_epochs])
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


#%% Do a round of fine-tuning of the entire model

# Unfreeze the base_model. Note that it keeps running in inference mode
# since we passed `training=False` when calling it. This means that
# the batchnorm layers will not update their batch statistics.
# This prevents the batchnorm layers from undoing all the training
# we've done so far.
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

base_model.trainable = True
model.summary()
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False
  
model.summary()

#%% Recompile the model (necessary for these changes to take effect), and resume training.
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(base_learning_rate/10),  # Low learning rate
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         validation_data=validation_dataset)


#%%
acc += history_fine.history['binary_accuracy']
val_acc += history_fine.history['val_binary_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.5, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


#%% Evaluation and prediction
loss, accuracy = model.evaluate(validation_dataset)
print('Test accuracy :', accuracy)


y_true = []
y_pred_probs = []
for images, labels in validation_dataset:
    y_true.extend(labels.numpy())
    y_pred_probs.extend(tf.sigmoid(model(images)).numpy())
    
# In a four-class problem, you might want to use argmax to get the predicted class
y_pred = np.argmax(y_pred_probs, axis=1)


#%%
'混淆矩陣'
CM = confusion_matrix(y_true, y_pred)
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 
print(accuracy(CM))
#%%
'混淆矩陣視覺化，看錯誤'
ax = plt.axes()
sns.heatmap(CM,annot=True, 
           fmt='g',
           annot_kws={"size": 10}, 
          
           xticklabels=class_names, 
           yticklabels=class_names, ax = ax)
ax.set_title('Confusion matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.show()


