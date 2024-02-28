# SignSense


SignSense - Redefining Communication with Tech Magic! 
 
       This project contains two parts:
      
  👉 Part 1: Signs to Words and Sounds;
For this we use YOLO v7 and Mediapipe:
* YOLO V7: For training we choose 8 classes ('hello','welcome','world','sign','language','nice to','meet you','thank you') and for that we labeled 2400 images where 2160 images used for training and 240 images for validation.Get a copy of coco.yaml file as custom.ymal,Do some changes in the custom.yaml file,download the initial weight and trained our model using this weight by the use of 1000 epochs.After the completion of training we used the 'last.pt',which is the best weight for the model,for the testing process.'detect.py',an in built function for testing the model,by the use of this model we can predict the sign language.

* Mediapipe: As we all known mediapipe is basically for action detection.Here we you Mediapipe Holistic model--> Mediapipe Holistic is a computer vision model provided by Google's MediaPipe library. It is designed for holistic understanding of the human body, offering a range of functionalities for pose estimation, face detection, hand tracking, and more. The holistic model combines multiple sub-models to provide a comprehensive representation of the human body.
 here first we Extract landmarks as numpy array,then we collect keypoints for training and testing.Here we trained 7 actions,each action has 60 number of sequence,where the length of the frames will be 30.For model creation we use CNN model,trained using 500 epochs,this model saved as 'action.h5'.

(for text to speech convertion we use 'gtts'.'gtts' stands for "Google Text-to-Speech," and it is a Python library that allows you to easily convert text into speech using Google Text-to-Speech API. With gtts, we can generate speech in various languages and save it as an audio file. )

  👉 Part 2: Text to Sign Language ;
 Here also we have two part:
 * first part is by the use of a website called fun translation,its free API allow us to translate text to ASL(American Sign Language).
 
 * second part is converting text into captivating Sign Language animations.In this stage we introducing our Keyframe Animation for Everyone,"KAE",which is an animated 3D model for converting our sign language into actions.
 
 (KAE is build by the use of the website DeepMotion) 
 
 
 For implement all these together we create a website "SignSense"
