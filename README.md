# CarND-Term1-P2-TrafficSignClassifier
Self-Driving Car Engineer Nanodegree Program: Term 1 Project 2

## Step 0: Load The Data

The Github repository provided separate pickle files for training, validation and test data (cell 1). Hence there was no need to split data.

## Step 1: Dataset Summary & Exploration

### Basic Summary

I used basic numpy operations to list number of images in each data set, their shapes and number of unique traffic sign classes (cell 2).
- Number of training examples = 34799
- Number of validation examples = 4410
- Number of testing examples = 12630
- Image data shape = (32, 32, 3)
- Number of classes = 43

### Exploratory Visualization

This step (cell 3) revealed that the 43 traffic sign classes cover signs for speed limits varying from 20 to 120 km/hr, road condition signs (no entry, priority, bumpy, slippery, etc.), direction signs (left, right, roundabout, etc.), pedestrian and vehicle information signs. They come in different shapes and colors. Each image size is 32x32x3 pixels which isn’t a lot of pixels to play with. Thus the images are blurry. The histogram revealed that they are also distributed unevenly in the training set. These are some challenges for the deep learning model.

## Step 2: Design and Test a Model Architecture

### Preprocess the Data Set

I tried to overcome the image blurriness with the *gaussian* function which negates the Gaussian blur from the original image through a weighted average of all pixels. To compare model performance with grayscaled images, I created the *gray* function using cv2.COLOR_RGB2GRAY. Finally I created the norm function to *normalize* all pixels between -0.5 and 0.5 to speed up model convergence (cells 4 and 5).
I also generated additional images to overcome the concern of underrepresenting certain classes (cell 6). To generate these additional images, I randomly selected images within each class and rotated them by an arbitrary angle from the [-20, -15, -10, -5, 5, 10, 15, 20] array. I explored two alternatives:
1. Each class to have number of images at least equal to the mean of the number of images in every class in the original training set (809 images for this training set)
2. Each class to have number of images equal to the maximum number of images in a single class in the original training set (2010 images for this training set)
Cells 7 and 8 apply this preprocessing to all images and plot random before/after image samples.

### Model Architecture

I used the LeNet architecture with a change in the output shape for the final fully connected layer (cell 10). I later added two dropout layers to improve accuracy (explained later in this writeup). I stuck with a batch size of 128 which my machine was capable of handling. Most of my validation accuracy plots began stabilizing after 10 epochs and did not improve after 15 epochs (cell 9). This table summarizes my final model architecture.

Layer | Input shape | Output shape
----- | ----------- | ------------
Convolutional | 32x32x3 | 28x28x6
ReLU Activation | 28x28x6 | 28x28x6
Max Pooling | 28x28x6 | 14x14x6
Convolutional | 14x14x6 | 10x10x16
ReLU Activation | 10x10x16 | 10x10x16
Max Pooling | 10x10x16 | 5x5x16
Flatten | 5x5x16 | 400x1
Fully Connected | 400x1 | 120x1
ReLU Activation | 120x1 | 120x1
Dropout | 120x1 | 120x1
Fully Connected | 120x1 | 84x1
ReLU Activation | 84x1 | 84x1
Dropout | 84x1 | 84x1
Fully Connected | 84x1 | 43x1

### Train, Validate and Test the Model

The training pipeline (cell 12) involved calculating a cross entropy loss for the training set logits over each epoch and using an Adam optimizer to minimize this loss. I found a learning rate of 0.001 a suitable compromise between validation accuracy and processing time because I did not see accuracy increase by more than 0.5% with a lower learning rate. The training set was shuffled prior to training.

I began initially with the LeNet architecture, no image preprocessing and no additional data to get a validation accuracy of around 88%. Image sharpening the colored images helped boost accuracy by around 2%. Grayscaling the images reduced processing time significantly but lowered accuracy by up to 1% at times. That seemed to indicate that the model was identifying colors to classify signs.

As described above, I generated additional images to avoid training biased towards certain classes. Both methods that I considered performed similarly, yielding an increase of up to 2.5%.

Validation accuracy varied between 92% and 93% on every attempt probably due to my use of data shuffling. One of my runs did yield an accuracy of 93.1% but I was not able to repeat that result and considered it a fluke. At this stage, my test accuracy was in the region of 90-91%.

Before trying additional image preprocessing that would have taken me a while to code, I tried adding dropout layers in my model as suggested in the video lectures. I added one such layer with a probability of 0.5 after the first fully connected layer with some improvement. Adding an identical layer after the second fully connected layer raised both the validation and test accuracies by around 2%. Changing the dropout probabilities to 0.7 yielded a minor consistent improvement over multiple training attempts. A combination of both dropout layers with alternative 2 for generating additional images boosted both accuracies by up to 1%.

**Final validation and test accuracies were 95.7% and 94.1% respectively** (cells 14 and 15). I performed multiple training attempts to confirm that this performance was consistent. My best attempt yielded validation and test accuracies of 96.1% and 94.3% respectively.

## Step 3: Test a Model on New Images

I downloaded 12 images of traffic signs while trying to represent each type of challenge I mentioned in Exploratory Visualization. These signs are of different colors, numbers, orientations and background colors (cell 16). One sign is partially covered in snow too! I resized the original images to 32x32x3 pixels using an external image processing tool before feeding them into my preprocessing function and testing them (cell 17; dropout probability was 1 i.e. no dropout).

Cell 18 shows the predictions for each test image. 7 out of 12 images were identified correctly, giving an accuracy of 58.3% for this set of images. The trained model failed to identify any of the 3 speed limit signs correctly. This behavior was consistent over multiple attempts; only once did it identify the 30km/hr sign correctly. It also struggled with the “right turn ahead” sign which was at an angled perspective, making it seem like a straight arrow. The “slippery road” sign seemed to have too much detail in a 32x32x3 image for proper identification. According to the softmax probabilities bar charts (cell 19), the model was quite sure of the signs that it identified correctly.

My trained model achieved the minimum required accuracy but there are some areas for improvement:
- Better image sharpening would help enhance details
- Using larger input images would help preserve some details
- Image perspective transformation rather than image rotation to generate additional images would help identify a wider range of real world traffic signs
- Training the model to identify numbers in addition to shapes would help identify the speed limit signs accurately
