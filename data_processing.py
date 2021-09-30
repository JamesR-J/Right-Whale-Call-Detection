import glob, zipfile, os, shutil, time

import pandas as pd
import tensorflow as tf

from pydub import AudioSegment


local_zip  = './whale_data.zip' #zip file of whale call audio data
TRAIN_dir  = './data/train'  #path where the videos are located
main_dir   = './data/train/'
aiff_dir   = os.path.join(main_dir, '*.aiff')
new_path_1 = os.path.join(main_dir, 'Right_Whale')
new_path_2 = os.path.join(main_dir, 'No_Right_Whale')
csv_path   = './data/train.csv'


def extract_files(local_zip):
    """

    Parameters
    ----------
    local_zip : string
        file directory of zip file to unzip

    Returns
    -------
    None.

    """
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('./')
    zip_ref.close()


def convert2wav(main_dir):
    """

    Parameters
    ----------
    main_dir : string
        directory of files, these are then converted to wav using AudioSegment

    Returns
    -------
    None.

    """
    owd = os.getcwd()
    os.chdir(main_dir)
    for video in glob.glob('*.aiff'):
        wav_filename = os.path.splitext(os.path.basename(video))[0] + '.wav'
        AudioSegment.from_file(video).export(wav_filename, format='wav')
    os.chdir(owd)  
 
    
def delete_aiff(aiff_dir):
    """

    Parameters
    ----------
    aiff_dir : string
        directory of files containing audio clips, deletes the .aiff files

    Returns
    -------
    None.

    """
    files = glob.glob(aiff_dir)
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))


def create_folders(new_path_1, new_path_2):
    """

    Parameters
    ----------
    new_path_1 : string
        creates folder of one path type
    new_path_2 : string
        creates folder of another path type

    Returns
    -------
    None.

    """
    if not os.path.isdir(new_path_1):
        os.makedirs(new_path_1)
    if not os.path.isdir(new_path_2):
        os.makedirs(new_path_2)
       
        
def edit_csv(csv_path):
    """

    Parameters
    ----------
    csv_path : string
        directory path of csv file

    Returns
    -------
    col_one : numpy array
        extracts edited names from csv file
    col_two : numpy array
        extracts label of whether it is a right whale audio sample or not

    """    
    train_csv = pd.read_csv(csv_path)
    train_csv = train_csv.replace('aiff', 'wav', regex=True)
    col_one = train_csv['clip_name'].to_numpy()
    col_two = train_csv['label'].to_numpy()
    
    return col_one, col_two
        
  
def new_filenames(TRAIN_dir, col_one, col_two):
    """

    Parameters
    ----------
    TRAIN_dir : string
        directory for the training files
    col_one : numpy arrray
        list of file names in the training folder
    col_two : numpy array
        list of labels from training folder, adds files with a label of 1 to 
        the right whale folder, and label of 0 to the no right whale folder

    Returns
    -------
    None.

    """
    filenames = tf.io.gfile.glob(str(TRAIN_dir) + '/*.wav')
    new_filenames = []
    for i in filenames:
      new_filenames.append(os.path.basename(i))
      for j in new_filenames:
        for count,k in enumerate(col_one):
          if j == k and col_two[count] == 1:
            original = os.path.join(TRAIN_dir, j)
            target = os.path.join(TRAIN_dir, 'Right_Whale/', j)
            shutil.copyfile(original,target)
            new_filenames.clear()
          elif j == k:
            original = os.path.join(TRAIN_dir, j)
            target = os.path.join(TRAIN_dir, 'No_Right_Whale/', j)
            shutil.copyfile(original,target)
    
    
def main():
    start = time.time()
    extract_files(local_zip)
    convert2wav(main_dir)
    delete_aiff(aiff_dir)
    create_folders(new_path_1, new_path_2)
    col_one, col_two = edit_csv(csv_path)
    new_filenames(TRAIN_dir, col_one, col_two)
    
    end = time.time()

    print('Right Whales:', len(os.listdir(new_path_1)))
    print('No Right Whales:', len(os.listdir(new_path_2)))        
    print(f"Runtime of the program was {end - start}s")


if __name__ == '__main__':
    main()