# Mask-detection
A real-time facial mask detector implemented by training deep learning model with PyTorch Lightning. 

Detecting face masks from images can be achieved by training deep learning models to classify face images with and without masks. This task is actually a pipeline of two tasks: first we have to detect if a face is present in an image/frame or not. Second, if a face is detected, find out if the person is wearing a mask or not. 

For the first task, I have used MTCNN to detect human faces. There are other approaches as well, we can use a simple cascade classifier to achieve this task. For the second task, I used a pretrained mobilenet_v2 model by modifying and training its classifier layers to classifiy face images with/without masks. 

For the dataset, I collected some face images with masks from the internet and some from the RWMFD dataset. For the images of faces, I used 
real face images from the 'real and fake face' dataset. A mobilenet_v2 model was trained on 1700 images in total with PyTorch Lightning. 

The aim of this project was of course to implement a face mask detector but I wanted to give PyTorch Lightning a try. It definetely makes your PyTorch implementation more organised and neat but implementation of certain tasks may feel a bit complicated, at least for me. 
