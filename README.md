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


## Usage
1) Set up your Spotify API.
2) Change spotify_credentials.py to work with your credentials.
3) Adjust the username of the cache file on music_data.py, without indicating cache location spotipy often timed out for me (the normal token limit is 1hr unless it refreshes).
4) Also in music_data.py adjust which playlists the script will create recommendations from, this process takes a while to run! So I limited mine to a small amount, but if you're happy to wait do as many as you please.
5) Run music_data.py which will create two pickle files of the recommended songs and your playlist songs.
6) Download your data from garmin connect and place within the source folder.
7) Work through garmin_data_analysis.ipynb and check your values for Upper and Lower Cadence (SPM), adjust the offsetter if required.
8) Within model_comparison.ipynb adjust the names of your favourite playlists, these are the playlists that you will "rate 1" to train the model, I used playlists that I have run to before, or any playlist of upbeat songs that I can imagine would uplift my run.
9) Work through model_comparison.ipynb to decide which model works best with your data.
10) Adjust the code in playlist_creation.py for a different model, playlist name, etc if required.
11) Run playlist_creation.py and enjoy the results! Ideally it should help you achieve faster running times, or at least better running form!!!

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
* Improve ratings system of tracks. eg rate out of 10 rather than just binary
* Create my own recommendation system rather than relying on Spotipys .
* Since some songs have a changing tempo Spotify averages out this BPM range, this can lead to some wrong songs making it through the BPM cut! Ideally a way to figure out the mode BPM of a song would be beneficial. This is a fairly large issue for me as a lot of Hardcore (one of my fave genres) has lots of tempo breakdowns.
* Run more so I can get some more data, aswell as track if the playlist helps improve times!!
