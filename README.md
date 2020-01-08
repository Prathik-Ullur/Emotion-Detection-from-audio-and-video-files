# apm_project_emotion_detection

## Overview

This repository contains our Fall 2019 term project for MIS 281N Advanced Predictive Modeling as part of the MS Business Analytics curriculum at UT Austin. It pulls from data found in the Ryerson Audio-Visual Database of Emotional Speech and Song (see here: https://zenodo.org/record/1188976).

### Contibutors

- Saurabh Bodas
- Prathik Ullur
- Namita Ramesh
- David Owen
- Pooja Shah

## Problem Statement

Speech is the primary form of communication between humans, containing both verbal and non-verbal cues which drive conversation. As simple chatbots evolve into complex home and personal assistants like Amazon Alexa and Google Assistant, the ability to detect nuance in human-human interactions is important in making human-AI interactions more natural. Interactions with computing machinery has advanced by becoming increasingly 'chatty' these days: the aforementioned systems make our daily lives easier by their ability to have a simple, coherent dialogue. But do any of them truly notice our emotions and react to them like a human conversational partner would?

The opposite party’s emotional state is evaluated and re-evaluated constantly by the brain in human-human interactions, and the inability to make a solid determination can be a great source of stress and awkwardness in conversation. Unlike humans, machines currently lack the abilities to perceive and show emotions. When a gap in emotional comprehension manifests in a human-AI interaction, the result is instead frustration or a loss of faith in the services the machine can provide. Consider speech-recognition programs used in less relaxed environments than the home, like help centers. Customers already frustrated with some sub-par level of service are directed to an AI unequipped to consider their emotional state, which may lead the customer to request a human representative. This scenario completely defeats the purpose of using AI to lessen the workload on human representatives.

Emotion detection can have a wide range of applications beyond strict human-AI communication. In transportation, installing emotion recognition (ER) systems in vehicles can allow the system to take over if fatigue is detected. In the field of education, detecting emotions like frustration and stress can act as a good feedback system for online learning portals. ER has lucrative applications in automated customer service. 

## Data Set: The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)

https://zenodo.org/record/1188976

The RAVDESS contains 7356 files. Each file was rated 10 times on emotional validity, intensity, and genuineness. Ratings were provided by 247 individuals who were characteristic of untrained adult research participants from North America. A further set of 72 participants provided test-retest data. High levels of emotional validity, interrater reliability, and test-retest intra-rater reliability were reported.

The database contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions, and song contains calm, happy, sad, angry, and fearful emotions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression. All conditions are available in three modality formats: Audio-only (16bit, 48kHz .wav), Audio-Video (720p H.264, AAC 48kHz, .mp4), and Video-only (no sound).

### Labels

- Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
- Vocal channel (01 = speech, 02 = song).
- Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
- Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
- Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
- Repetition (01 = 1st repetition, 02 = 2nd repetition).
- Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).


## Approach

Emotions manifest in three primary forms: facial expressions, vocal inflections, and body language. We will focus on the first two, as they are more consistent across the entire human population and be more easily attributed to specific emotions.

The first step in any automatic speech recognition system is to extract features i.e. identify the components of the audio signal that are good for identifying the linguistic content and removing all the noise which carries information like background noise, emotion etc. Mel Frequency Cepstral Coefficents (MFCCs) are a feature widely used in automatic speech and speaker recognition. This process allows the computer to understand sound as humans do. It attempts to mimic the human cochlea (an organ in the ear) which vibrates at different spots depending on the frequency of the incoming sounds. We will use the Python package librosa for this purpose.

Other features that can be extracted are Zero Crossing Rate, Spectral Centroid and Spectral Rolloff.

- Zero Crossing Rate: The zero crossing rate is the rate of sign-changes along a signal, i.e., the rate at which the signal changes from positive to negative or back.

- Spectral Centroid: It indicates where the ”centre of mass” for a sound is located and is calculated as the weighted mean of the frequencies present in the sound.

- Spectral Rolloff: Spectral rolloff is the frequency below which a specified percentage of the total spectral energy, e.g. 85%, lies.


### Modelling Approach

We have run different models like XGBoost, Support Vector Machines, Convolution Neural Network, and Recurrent Neural Networks on the existing and manually engineered features present in audio and video information.

Since our dataset has only ~7000 observations, we could find a pretrained model on a similar problem and then implement transfer learning. We will also look at several audio augmentation techniques like Changing Pitch, Noise Injection and Shifting Time to expand our dataset.

### Usage

Code is contained in ipython files meant to run in Google Colab, but they can also be run on a local machine with the source data in the appropriate directories. Users can begin at the "cnn.load_model()" cell to skip training and writing to a new .h5 file, which will take anywhere from 45min to several hours.

Trained CNN model weights can be stored in .h5 files created by the Keras Python package. These files can vary in size from 2-800 Mb, far beyond the storage limits of Git. Instead, they can be found at this dropbox link: https://www.dropbox.com/sh/29soflhyqra5qfi/AACE6L0D6ygSDDYq3c9syYLga?dl=0
