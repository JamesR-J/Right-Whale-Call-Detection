# Right-Whale-Call-Detection
Model to predict whether an audio sample contains a North Atlantic Right Whale call

talk about using kaggle dataset so had to preprocess cus in .aiff and that tf function uses wav
also since was doing this, wasn't hard to put files into folders so could use flow from direcotry rather than flow from dataframe, literally no reason just was easy
then say provided my processed dataset too for ease for some people, ie the processing took like 48hrs or something stupid
talk about no labels on test data so it was unusable basically
mention based on the tensorflow speech recognition walktrhough

future plans:
maybe a way to detet diff whales like personal id
improve the model for further performance
try it for other whale call datasets, thers a minke one to look at 

as my first project i leavees a lot to be desired but was a good step in beginning to understand how to use tensorflow and other things


## Usage
1) Either download the Kaggle Dataset and use data_processing.py to convert the .aiff files into .wav. Or download the compressed data folder and use that.
2) Work through XXX

## Requirements to run:
* Spotify account
* Garmin Connect account
* Install the required libraries below

## Libraries:
* pandas
* AudioSegment
* Tensorflow
* Keras
* natsort
* Seaborn

## Next Steps:
* ALthough it currently works well it needs a whole redesign to properly work, ie using binary calssification etc etc
* try about with masking and preprocessing of the files to enable better accuracy
* more hyper paramater tuning to optimise the process
