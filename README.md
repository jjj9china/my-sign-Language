# my-sign-Language
1.	Use build_data_set.py to build your dataset. 

1.1	Before doing that you should record a video and show your gestures in a contrasting background. For example, if your hand is light-colored, you should find a background with a dark area as your background; if your hand is dark, you are ready to find a light background.

1.2	After you have a video, first thing you should do is find the area where your hand in every image. So this time you may use the find_best_area.py to help you decide it. Try several times until your hand appears in the center of the box. The box should not be too big. 

1.3	Now you can run build_data_set.py to add one gesture in your dataset .Note to change the file path when you reading a video and write a image in line 18 and line 34.

2.	And now we need do is split the data set into a training set and a test set. And our code load_images.py implement this function. After you run this code , you can get train_images, train_labels ,test_images, test_labels in the current folder.

3.	In cnn_keras.py we build our CNN, and its structure have recorded in the model.png . We saved our model during training ,and that is cnn_model_keras2.h5. By the way, we use keras as our CNN platform.

4.	After training CNN, what we need to do next is to read gestures on the video screen and output the predictions in real time. These functions are implemented in recognize_gestures.py . In this code you need to use find_best_area.py to help you find best gesture areas as well .

4.1	We first read each frame of the video in the video screen, then collect specific gesture areas and preprocess them (as we did with the data set). 

4.2	Then put the processed pictures into the network for training. The CNN network will return the largest output category and the corresponding probability.

4.3	If the probability is greater than 80%, we believe that the network outputs the correct result and displays the letters corresponding to the output category on the screen.

In addition to the above-mentioned process, there are many details that need attention in this project. Only you will find it in the process of realizing it. It is not covered here. I hope this project will help you.

The main reference for this work comes from https://github.com/EvilPort2/Sign-Language  
