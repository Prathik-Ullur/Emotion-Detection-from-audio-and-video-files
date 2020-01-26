
# How to detect emotion from audio and video files?

![](https://marketingland.com/wp-content/ml-loads/2016/03/emotions_ss_1920.png)

It all began about 2,000 years ago when Plato wrote, “All learning has an emotional base.”

One might think that over 200,000 years of evolution would make humans masters of emotions. Yet, we live in a world where people, irrespective of age or maturity, often make errors in emotional judgment. Clarity in identifying emotions is key to social behaviors such as smooth communication and building long-lasting relationships.

What makes identifying emotions challenging for humans?

We often struggle to express our emotions and articulate our feelings. emotions come in many different degrees, qualities, and intensities. In addition, our experiences are often comprised of multiple emotions at once, which adds another dimension of complexity to our emotional experience.

The icing on the cake is, however, Emotional Bias. With a spectrum as variant as the range of emotions, there is bound to be bias. This is where the problem gets interesting for us as data scientists- we love a good ‘Bias-Variance’ problem!

Enter, your friendly, unbiased neighborhood Emotion Detector Bot. Gone are the days when the only thing separating man and machine was emotional intelligence. Emotion Recognition or Artificial ‘Emotional’ Intelligence is now a $20 billion field of research with applications in many different industries.

Across industries, artificial emotional intelligence can work in a number of ways. For example, AI can monitor a user’s emotions and analyze them to achieve a certain outcome. This application would prove extremely useful in enhancing automated Customer Service calls. AI can also use emotional readings as part of decision making, for example in marketing campaigns. Advertisements can be changed based on consumer reactions.

This repository contains our Fall 2019 term project for MIS 281N Advanced Predictive Modeling as part of the MS Business Analytics curriculum at UT Austin. It pulls from data found in the Ryerson Audio-Visual Database of Emotional Speech and Song (see here: https://zenodo.org/record/1188976).



## Problem Statement

Speech is the primary form of communication between humans, containing both verbal and non-verbal cues which drive conversation. As simple chatbots evolve into complex home and personal assistants like Amazon Alexa and Google Assistant, the ability to detect nuance in human-human interactions is important in making human-AI interactions more natural. Interactions with computing machinery has advanced by becoming increasingly 'chatty' these days: the aforementioned systems make our daily lives easier by their ability to have a simple, coherent dialogue. But do any of them truly notice our emotions and react to them like a human conversational partner would?

The opposite party’s emotional state is evaluated and re-evaluated constantly by the brain in human-human interactions, and the inability to make a solid determination can be a great source of stress and awkwardness in conversation. Unlike humans, machines currently lack the abilities to perceive and show emotions. When a gap in emotional comprehension manifests in a human-AI interaction, the result is instead frustration or a loss of faith in the services the machine can provide. Consider speech-recognition programs used in less relaxed environments than the home, like help centers. Customers already frustrated with some sub-par level of service are directed to an AI unequipped to consider their emotional state, which may lead the customer to request a human representative. This scenario completely defeats the purpose of using AI to lessen the workload on human representatives.

Emotion detection can have a wide range of applications beyond strict human-AI communication. In transportation, installing emotion recognition (ER) systems in vehicles can allow the system to take over if fatigue is detected. In the field of education, detecting emotions like frustration and stress can act as a good feedback system for online learning portals. ER has lucrative applications in automated customer service. 


## RAVDESS Data Set

The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) contains 7356 files (total size: 24.8 GB). The database contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression. All conditions are available in three modality formats: Audio-only (16bit, 48kHz .wav), Audio-Video (720p H.264, AAC 48kHz, .mp4), and Video-only (no sound).

Each of the 7356 RAVDESS files has a unique filename. The filename consists of a 7-part numerical identifier (e.g., 02–01–06–01–02–01–12.mp4). These identifiers define the stimulus characteristics:

Filename identifiers:

•	Modality (01 = full-AV, 02 = video-only, 03 = audio-only)

•	Vocal channel (01 = speech, 02 = song).

•	Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).

•	Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the ‘neutral’ emotion.

•	Statement (01 = “Kids are talking by the door”, 02 = “Dogs are sitting by the door”).

•	Repetition (01 = 1st repetition, 02 = 2nd repetition).

•	Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).


## Approach

Emotions manifest in three primary forms: facial expressions, vocal inflections, and body language. We will focus on the first two, as they are more consistent across the entire human population and be more easily attributed to specific emotions.

### Speech :

The first step in any automatic speech recognition system is to extract features i.e. identify the components of the audio signal that are good for identifying the linguistic content and removing all the noise which carries information like background noise, emotion etc. Mel Frequency Cepstral Coefficents (MFCCs) are a feature widely used in automatic speech and speaker recognition. This process allows the computer to understand sound as humans do. It attempts to mimic the human cochlea (an organ in the ear) which vibrates at different spots depending on the frequency of the incoming sounds. We will use the Python package librosa for this purpose.

Other features that can be extracted are Zero Crossing Rate, Spectral Centroid and Spectral Rolloff.

- Zero Crossing Rate: The zero crossing rate is the rate of sign-changes along a signal, i.e., the rate at which the signal changes from positive to negative or back.

- Spectral Centroid: It indicates where the ”centre of mass” for a sound is located and is calculated as the weighted mean of the frequencies present in the sound.

- Spectral Rolloff: Spectral rolloff is the frequency below which a specified percentage of the total spectral energy, e.g. 85%, lies.

### Video :

The first step would be to parse the video file into a set of image frames that we use to train the model. We use the cv2 library to capture images from a video. The VideoCapture function reads the video file and converts it into a sequence of image frames.

Each frame obtained will contain a two-dimensional array of integers containing information of the image. The images are composed of pixels and these pixels are channels of multiple arrays of numbers. Colored images have three color channels — red, green, and blue — and each channel is represented by a grid. Each cell in the grid stores a number between 0 and 255 which denotes the intensity of that cell. To capture a different expression each time, we pass every 20th frame into our training model.

After extracting the image data, we resize the images to 256*256 to retain as much information as possible to enhance the accuracy of the model. We converted these images to gray-scale, so that there is only one channel thereby reducing complexity.

## Modelling Approach

We have run different models like XGBoost, Support Vector Machines, Convolution Neural Network, and Recurrent Neural Networks on the existing and manually engineered features present in audio and video information.

Since our dataset has only ~7000 observations, we could find a pretrained model on a similar problem and then implement transfer learning. We will also look at several audio augmentation techniques like Changing Pitch, Noise Injection and Shifting Time to expand our dataset.

## Usage

Code is contained in ipython files meant to run in Google Colab, but they can also be run on a local machine with the source data in the appropriate directories. Users can begin at the "cnn.load_model()" cell to skip training and writing to a new .h5 file, which will take anywhere from 45min to several hours.

Trained CNN model weights can be stored in .h5 files created by the Keras Python package. These files can vary in size from 2-800 Mb, far beyond the storage limits of Git. Instead, they can be found at this dropbox link: https://www.dropbox.com/sh/29soflhyqra5qfi/AACE6L0D6ygSDDYq3c9syYLga?dl=0

## Challenges and Future Scope

### Combining audio & video data:

This was undoubtedly our biggest challenge in this project. So far, we have split our data into separate audio and video files to extract MFCCs and images respectively. However, taking a combined approach to simultaneously train a model capable of processing audio and video signals would help achieve a more scalable outcome. As mentioned earlier in the blog, emotion recognition is majorly sought after in many industries. We believe as future scope, this product could be made more widely acceptable by being compatible with either kind of input.

### Emotion recognition in Health Care:

An industry that’s taking advantage of this technology currently is Health Care, with AI-powered recognition software helping to decide when patients necessitate medicine or to help physicians determine who to see first. A problem we foresee that can be prevented with accurate emotion detection is in the Mental Health awareness space. Those suffering from mental health issues often keep to themselves and don’t share much about their problems. Correctly identifying emotions these distress signals, could make a huge difference to avoid mental breakdowns and stress-related trauma. A computer would be unbiased and more sensitive to detecting early signs to help alert close friends or family.

### Dealing with the inherent bias:

There are two broad biases that are suffered by our models:

1. All the actors are from the North American geographic area, and thus speak in a distinct North American accent, causing our models to be biased to that. Audio data from speakers of other geographic locations would help eliminate this bias.

2. All audio and video recordings are taken in a professional setting at Ryerson University in the absence of any background/white noise. Therefore, models that are trained on this dataset may not perform well on real-world data. A potential fix to this situation could be to train models on noisy audiovisual datasets and attach class labels using the Amazon Turk service.

## References:

[1] Muneeb ul Hassan, VGG16 — Convolutional Network for Classification and Detection

[2] Sourish Dey, CNN application on structured data-Automated Feature Extraction

[3] Francesco Pochetti, Video Classification Experiments: combining Image with Audio features

[4] Ryan Thompson, How to Use Google Colaboratory for Video Processing

[5] James Lyons, Python Speech Features

[6] Angelica Perez, EmoPy: a machine learning toolkit for emotional expression

## [Link to Towards Data Science Blog](https://towardsdatascience.com/aei-artificial-emotional-intelligence-ea3667d8ece)


