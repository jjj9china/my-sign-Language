import cv2, os
import numpy as np
import random
from sklearn.utils import shuffle
import pickle

def pickle_images_labels():
	gest_folder = "my_gestures"
	images_labels = []
	images = []
	labels = []
	for g_id in os.listdir(gest_folder):  # 返回指定的文件夹包含的文件或文件夹的名字的列表
		for i in range(100):
			img = cv2.imread(gest_folder+"/"+g_id+"/"+str(i+1)+".jpg", 0)
			if np.any(img == None):
				continue
			images_labels.append((np.array(img, dtype=np.float32), int(g_id))) # label就是文件夹名称0~44
	return images_labels

def split_images_labels(images_labels):
	images = []
	labels = []
	for (image, label) in images_labels:
		images.append(image)
		labels.append(label)
	return images, labels

# 获取所有图像和标签
images_labels = pickle_images_labels()
# 打乱顺序
images_labels = shuffle(shuffle(shuffle(images_labels)))  # type=unit8,[(Numpy array,0),(Numpy array,11)...(Numpy array,37)]
# 分离图像和标签
images, labels = split_images_labels(images_labels)
print("Length of images_labels", len(images_labels))

train_images = images[:int(5/6*len(images))] # 前5/6张图片作为训练集
print("Length of train_images", len(train_images))
with open("train_images", "wb") as f:
	pickle.dump(train_images, f)
del train_images

train_labels = labels[:int(5/6*len(labels))]
print("Length of train_labels", len(train_labels))
with open("train_labels", "wb") as f:
	pickle.dump(train_labels, f)
del train_labels

test_images = images[int(5/6*len(images)):] # 后1/6张图片作为测试集
print("Length of test_images", len(test_images))
with open("test_images", "wb") as f:
	pickle.dump(test_images, f)
del test_images

test_labels = labels[int(5/6*len(labels)):]
print("Length of test_labels", len(test_labels))
with open("test_labels", "wb") as f:
	pickle.dump(test_labels, f)
del test_labels