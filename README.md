# Right-Whale-Call-Detection
Model to predict whether an audio sample contains a North Atlantic Right Whale call.

This is one of my first projects so it leaves a lot to be desired but it was a good step in understanding the basics of Tensorflow! It is heavily based off the [Tensorflow Simple Audio Recognition](https://www.tensorflow.org/tutorials/audio/simple_audio) guide but applied to recognise whale calls using the [Kaggle Right Whale Dataset](https://www.kaggle.com/c/whale-detection-challenge/data).

Due to the Kaggle dataset using .aiff files and the Tensorflow primarily based in .wav files, all the files had to be converted. The data_processing.py was used to do this but it takes a very long time to run (think around 24hr)! I also split the files into subfolders so that flow_from_directory could be used, although this wasn't a necessary step since flow_from_dataframe could also work. With this dataset there is also a test folder but since there are no labels for them it wasn't very useful.

Although this model produces a good accuracy it is definitely a very limited entry point into the field. Some further improvements I aim to work on are listed below:
* Further model refinement and hyper-parameter tuning.
* Remodelling the functions and Neural Network to be binary based since it is just a yes or no problem, currently it is working as a categorical problem.
* See if the model translates to other datasets, there is a relevent [Minke Whale Dataset](http://www.soest.hawaii.edu/ore/dclde/dataset/).
* Try some masking and preprocessing of files to enable better accuracy as seen in the [winning solution](https://github.com/jaimeps/whale-sound-classification).


## Usage
1) Download the Kaggle Dataset and use data_processing.py to convert the .aiff files into .wav
2) Work through whale_detection.ipynb

## Libraries:
* pandas
* AudioSegment
* Tensorflow
* Keras
* natsort
* Seaborn
