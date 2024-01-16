
#批量tiff转jg
#代码中路径更改为自己图像存放路径即可
from PIL import Image
import os

imagesDirectory = r"C:\Users\user\Desktop\Python_Practice\CNN\Eardrum\tympanoskleros"# tiff图片所在文件夹路径
distDirectory = r"C:\Users\user\Desktop\Python_Practice\CNN\Eardrum\tympanoskleros_new"# 要存放jpg格式的文件夹路径
for imageName in os.listdir(imagesDirectory):
    imagepath = os.path.join(imagesDirectory, imageName)
    image = Image.open(imagepath)# 打开tiff图像
    distImagepath= os.path.join(distDirectory,imageName[:-4]+'.jpg')#更改图像后缀为·jpg,并保证与原图像名
    print(imagepath)
    image.save(distImagepath)#保存jpg图像