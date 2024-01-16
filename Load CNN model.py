import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt   
#%%
model=tf.keras.applications.xception.Xception(weights='imagenet',include_top=True)

#%%
'任意照片&格式轉換'
IMAGE_PATH=r'C:\Users\user\Desktop\Python_Practice\CNN\Eardrum_new\Normal\normal_1..jpg'
img=tf.keras.preprocessing.image.load_img(IMAGE_PATH,target_size=(299,299))
img.show()
img=tf.keras.preprocessing.image.img_to_array(img)
#%%
'辨識'
predictions=model.predict(np.array([img]))
print('Predicted:', tf.keras.applications.xception.decode_predictions(predictions, top=3)[0])