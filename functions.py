import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from IPython import display
from tensorflow.keras.layers.experimental import preprocessing


AUTOTUNE = tf.data.experimental.AUTOTUNE

def data_info(dir1, dir2):
    """

    Parameters
    ----------
    dir1 : string
        directory of no right whale folder
    dir2 : string
        directory of right whale folder

    Returns
    -------
    labels : list
        list of labels indicating if right whale call or not in audio sample
    filenames : numpy array
        array of all the data files in wav form
    num_samples : int
        number of total samples

    """
    
    labelers = np.array(tf.io.gfile.listdir(str(dir1)))
    wanted_labels = ['No_Right_Whale', 'Right_Whale']
    labels = [x for x in labelers if x in wanted_labels]
    labels = np.array(labels)
    filenames = tf.io.gfile.glob(str(dir1) + '/*/*')
    filenames = tf.random.shuffle(filenames) #shuffles dataset
    num_samples = len(filenames)
    print('No Right Whale Call Audio Sample:', 
          len(os.listdir(os.path.join(dir2, 'No_Right_Whale'))))
    print('Right Whale Call Audio Sample:   ', 
          len(os.listdir(os.path.join(dir2, 'Right_Whale'))))
    print('Sample Labels:', labels)
    print('Number of total examples:        ', num_samples)
    print('Example file tensor:', filenames[0])
    
    return labels, filenames, num_samples


def test_train_split(samples, filenames):
    """

    Parameters
    ----------
    samples : int
        number of audio samples
    filenames : numpy array
        array of all the data files in wav form

    Returns
    -------
    train_files : numpy array
        80% split for training data from all files
    val_files : numpy array
        10% split for validation data from all files
    test_files : numpy array
        10% split for testing data from all files

    """
    a1 = samples/100
    lowerlimit = int(a1 * 80)
    upperlimit = int(a1 * 10)
    train_files = filenames[:lowerlimit]  #80/10/10 split 
    val_files = filenames[lowerlimit: samples-upperlimit]
    test_files = filenames[-upperlimit:]

    print('Training set size  ', len(train_files))
    print('Validation set size ', len(val_files))
    print('Test set size       ', len(test_files))
    
    return train_files, val_files, test_files


def decode_audio(audio_binary):
    """

    Parameters
    ----------
    audio_binary : nmupy array
        a wav file

    Returns
    -------
    TYPE
        converts and normalises the wav into a float tensor 

    """
    audio, _ = tf.audio.decode_wav(audio_binary)
    
    return tf.squeeze(audio, axis=-1)


def get_waveform_and_label(file_path):
    """

    Parameters
    ----------
    file_path : string
        directory of a file in the dataset

    Returns
    -------
    waveform : float tensor
        contains the information from the wav file from the directory
    label : string
        either indicates sample has a right whale in it or not

    """
    label = tf.strings.split(file_path, os.path.sep)[-2]
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    
    return waveform, label


def create_waveform(files):
    """

    Parameters
    ----------
    files : numpy array
        an array of data from all the training files

    Returns
    -------
    waveform_ds : tensorflow dataset
        dataset from the training files converted to a float tensor

    """
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    waveform_ds = files_ds.map(get_waveform_and_label, 
                               num_parallel_calls=AUTOTUNE)
    
    return waveform_ds#, files_ds


def get_spectrogram(waveform):
    """

    Parameters
    ----------
    waveform : float tensor
        a file from the tensorflow dataset of the audio files

    Returns
    -------
    spectrogram : float tensor
        computes the short time fourier transform (stft) of the file

    """
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
    #pads audio to make sure all samples will be the same length
  
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram) #converts the complex tensor from stft
    #into a flaot tensor

    return spectrogram


def audio_playback(waveform_ds):
    """

    Parameters
    ----------
    waveform_ds : tensorflow dataset
        dataset from the training files converted to a float tensor

    Returns
    -------
    label : string
        label of the file indicating if right whale or not
    spectrogram : float tensor
        computes the short time fourier transform (stft) of the file

    """
    for waveform, label in waveform_ds.take(1):
        label = label.numpy().decode('utf-8')
        spectrogram = get_spectrogram(waveform)

    print('Label:', label)
    print('Waveform shape:', waveform.shape)
    print('Spectrogram shape:', spectrogram.shape)
    print('Audio playback')
    display.display(display.Audio(waveform, rate=16000))
    
    return label, spectrogram


def plot_spectrogram(spectrogram, ax):
    """

    Parameters
    ----------
    spectrogram : float tensor
        computes the stft of the file
    ax : matplotlib.pyplot axis
        axis value for subplots

    Returns
    -------
    None.

    """
    log_spec = np.log(spectrogram.T)
    #convert to frequencies to log scale and transpose so that the time is
    #represented in the x-axis (columns). leads to an error to ignore about
    #dividing by 0 encountered in log
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)
    
    
def get_spectrogram_and_label_id(audio, label):
    """

    Parameters
    ----------
    audio : float tensor
        audio float tensor from the waveform dataset
    label : string
        labels from the waveform dataset

    Returns
    -------
    spectrogram : float tensor
        converts the audio format into a spectrogram
    label_id : int
        values to describe the string labels

    """
    labels = ['No_Right_Whale', 'Right_Whale']
    labels = np.array(labels)
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == labels)
    
    return spectrogram, label_id


def create_spectrogram(files):
    """

    Parameters
    ----------
    files : numpy array
        files from the dataset

    Returns
    -------
    spectrogram_ds : tensorflow dataset
        files from the training split converted by stft into spectrograms

    """
    waveform_ds = create_waveform(files)
    spectrogram_ds = waveform_ds.map(get_spectrogram_and_label_id, 
                                     num_parallel_calls=AUTOTUNE)
    
    return spectrogram_ds


def preprocess_dataset(files):
    """

    Parameters
    ----------
    files : numpy array
        files from the dataset

    Returns
    -------
    output_ds : tensorflow dataset
        files from the test/validation split converted by stft into 
        spectrograms

    """
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, 
                             num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
      get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
    
    return output_ds


def NN_preprocessing(spectrogram_ds, labels):
    """

    Parameters
    ----------
    spectrogram_ds : tensorflow dataset
        dataset from the training files converted to a float tensor via short
        time fourier transform
    labels : list
        list of labels corresponding to the files in the dataset

    Returns
    -------
    input_shape : numpy array (x, x, x)
        first two values indicate size of image, last value is for colour, 1 
        for grayscale, 3 for rgb
    norm_layer : tensorflow.keras.layers
        normalisation layer for files fed from the dataset
    num_labels : int
        number of labels of dataset, for this current problem it is 2

    """
    for spectrogram, _ in spectrogram_ds.take(1):
        input_shape = spectrogram.shape
    print('Input shape:', input_shape)
    num_labels = len(labels)
    print('Number of Labels:', num_labels)

    norm_layer = preprocessing.Normalization()
    norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))
    return input_shape, norm_layer, num_labels


def NN_graph(label1, label2, values, col1, col2):
    """

    Parameters
    ----------
    label1 : string
        data type to use for first graph
    label2 : string
        data type to use for second graph
    values : tensorflow metrics
        data from the model on accuracy, loss, etc
    col1 : string
        colour of training set line
    col2 : string
        colour of validation set line

    Returns
    -------
    None.

    """
    acc = values.history['accuracy']
    val_acc = values.history['val_accuracy']
    loss = values.history['loss']
    val_loss = values.history['val_loss']
    x = values.epoch

    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    ax1.plot(x, acc, col1, label ='Training ' + label1)
    ax1.plot(x, val_acc, col2, label='Validation ' + label1)
    ax1.set_title('Training and Validation ' + label1)
    ax1.set_ylim([0.85, 1])
    ax1.legend(loc=0)
    ax2.plot(x, loss, col1, label='Training ' + label2)
    ax2.plot(x, val_loss, col2, label='Validation ' + label2)
    ax2.set_title('Training and Validation ' + label2)
    ax2.legend(loc=0)
    fig.tight_layout()