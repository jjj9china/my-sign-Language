import cv2, pickle
import numpy as np
import tensorflow as tf
#from cnn_tf import cnn_model_fn
import os
import sqlite3
import time
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
classifier = tf.estimator.Estimator(model_dir="tmp/cnn_model2", model_fn=cnn_model_fn)
prediction = None
model = load_model('cnn_model_keras2.h5')

def get_image_size():
	img = cv2.imread('gestures/0/100.jpg', 0)
	return img.shape

image_x, image_y = get_image_size()

def tf_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	np_array = np.array(img)
	return np_array

def tf_predict(classifier, image):
	'''
	need help with prediction using tensorflow
	'''
	global prediction
	processed_array = tf_process_image(image)
	pred_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":processed_array}, shuffle=False)
	pred = classifier.predict(input_fn=pred_input_fn)
	prediction = next(pred)
	print(prediction)

def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

def get_pred_text_from_db(pred_class):
    if pred_class==0:
        return "none"
    if pred_class==1:
        return "S" 
    if pred_class==2:
        return "J"
    if pred_class==3:
        return "T"
    if pred_class==4:
        return "U"
    """
    conn = sqlite3.connect("gesture_db.db")
    cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
    cursor = conn.execute(cmd)
    for row in cursor:
        return row[0]
    """

def split_sentence(text, num_of_words):
	'''
	Splits a text into group of num_of_words
	'''
	list_words = text.split(" ")
	length = len(list_words)
	splitted_sentence = []
	b_index = 0
	e_index = num_of_words
	while length > 0:
		part = ""
		for word in list_words[b_index:e_index]:
			part = part + " " + word
		splitted_sentence.append(part)
		b_index += num_of_words
		e_index += num_of_words
		length -= num_of_words
	return splitted_sentence

def put_splitted_text_in_blackboard(blackboard, splitted_text):
	y = 100
	for text in splitted_text:
		cv2.putText(blackboard, text, (4, y), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255))
		y += 10

def get_hand_hist():
	with open("hist", "rb") as f:
		hist = pickle.load(f)
	return hist


   
global prediction
sentence = []    
save_img= 0
cap = cv2.VideoCapture('test.mp4')  
while(cap.isOpened()):
    text=""
    ret, frame = cap.read()  
    hf, wf, _ = frame.shape 
    img =frame[130:300, 530:700]
    h, w, _ = img.shape  
  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    gray=cv2.blur(gray,(5,5))
    ret, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY) 
      
    # Find Contour  
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  
      
    # 需要搞一个list给cv2.drawContours()才行！！！！！  
    c_max = []  
    for i in range(len(contours)):  
        cnt = contours[i]  
        area = cv2.contourArea(cnt)  
          
        # 处理掉小的轮廓区域，这个区域的大小自己定义。  
        if(area < (h/10*w/10)):  
            c_min = []  
            c_min.append(cnt)  
            # thickness不为-1时，表示画轮廓线，thickness的值表示线的宽度。  
            cv2.drawContours(thresh, c_min, -1, (0,0,0), thickness=-1)  
            continue    
        c_max.append(cnt)  
      
    cv2.drawContours(thresh, c_max, -1, (255, 255, 255), thickness=-1)  
    save_img = thresh 
    
    pred_probab, pred_class = keras_predict(model, save_img)
    cv2.imshow("save_img",save_img)
    print(pred_class, pred_probab)
    
    
    text = get_pred_text_from_db(pred_class)
    if not text in sentence:
        sentence.append(text)
    print(text)
    
    blackboard = np.zeros((hf, 300, 3), dtype=np.uint8)
    splitted_text = split_sentence(text, 2)
    put_splitted_text_in_blackboard(blackboard, splitted_text)
    cv2.putText(blackboard, "".join(sentence), (30, 200), cv2.FONT_HERSHEY_TRIPLEX, 1.3, (255, 255, 255))
    cv2.rectangle(frame, (530,130), (700,300), (0,255,0), 2)
    
    res = np.hstack((frame, blackboard))
    cv2.imshow("Recognizing gesture", res)
    
    #cv2.imshow("thresh", thresh)
    time.sleep(0.1)
    if cv2.waitKey(1) == ord('q'):
        #cam.release()
        cv2.destroyAllWindows()
        break
    
