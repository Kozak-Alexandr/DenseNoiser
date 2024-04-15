import tensorflow as tf
from tensorflow import keras
#from keras.optimisers import Adam
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, Multiply, Concatenate, Conv1D, GRU, MaxPooling1D, Flatten, BatchNormalization, Activation, Reshape
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import random
from tensorflow.python.client import device_lib

#GPU memory smol proceed with CPU + RAM
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model_name = "DenseNetNoise"
block_length = 0.050  
voice_max_length = int(0.5 / block_length)  
frame_length = 512
batch_size = 2048
epochs = 2

#learning_rate = 0.003#default is 0.001 ---> too long ad seems to be stuck
optimiser = tf.keras.optimizers.Adam()
optimiser.learning_rate.assign(0.003)

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

print("voice_max_length:", voice_max_length)

def audioToTensor(filepath: str):
    """
    Reads a WAV file, converts it to a spectrogram, and returns the components.

    Args:
        filepath (str): Path to the WAV file.

    Returns:
        tuple: A tuple containing:
            - spect_real (tf.Tensor): Real part of the spectrogram.
            - spect_image (tf.Tensor): Imaginary part of the spectrogram.
            - spect_sign (tf.Tensor): Sign of the real part of the spectrogram.
            - audioSR (int): Sample rate of the audio.
    """
    audio_binary = tf.io.read_file(filepath)
    audio, audioSR = tf.audio.decode_wav(audio_binary)
    audioSR = tf.get_static_value(audioSR)
    audio = tf.squeeze(audio, axis=-1)
    # 16000*0.008=128
    frame_step = int(audioSR * 0.008)  
    spectrogram = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step)
    spect_image = tf.math.imag(spectrogram)
    spect_real = tf.math.real(spectrogram)
    spect_sign = tf.sign(spect_real)
    spect_real = tf.abs(spect_real)
    return spect_real, spect_image, spect_sign, audioSR

def spectToOscillo(spect_real, spect_sign, spect_image, audioSR):
    """
    Reconstructs an audio waveform from a spectrogram.

    Args:
        spect_real (tf.Tensor): Real part of the spectrogram.
        spect_image (tf.Tensor): Imaginary part of the spectrogram.
        spect_sign (tf.Tensor): Sign of the real part of the spectrogram.
        audioSR (int): Sample rate of the audio.

    Returns:
        tf.Tensor: The reconstructed audio waveform.
    """
    frame_step = int(audioSR * 0.008)
    # spect_real = pow(10, spect_real/20)#power value
    spect_real *= spect_sign
    spect_all = tf.complex(spect_real, spect_image)
    inverse_stft = tf.signal.inverse_stft(spect_all, frame_length=frame_length, frame_step=frame_step,
                                          window_fn=tf.signal.inverse_stft_window_fn(frame_step))
    return inverse_stft

class MySequence(tf.keras.utils.Sequence):
    """
    Custom data generator for training the noise reduction model.

    This class iterates over the audio files, preprocesses them into spectrograms,
    and generates batches of noisy and clean spectrograms for training.
    """
    def __init__(self, x_train, x_train_count, batch_size, is_noisy=True):
        self.x_train = x_train
        self.x_train_count = x_train_count
        self.batch_size = batch_size
        self.is_noisy = is_noisy

    def __len__(self):
        """
        Returns the number of batches in the dataset.
        """
        return self.x_train_count // self.batch_size

    def __getitem__(self, idx):
        batch_x_train = np.zeros((batch_size, int(frame_length/2+1)))
        batch_y_train = np.zeros((batch_size, int(frame_length/2+1)))
        current_size = 0
        while current_size < batch_size:
            path_clear, _ = self.x_train[(idx * self.batch_size + current_size)%len(clear_files)]
            path_noisy = path_clear.replace("clear", "noisy")
            spectNoisy, _, _, _ = audioToTensor(path_noisy)
            spectClear, _, _, _ = audioToTensor(path_clear) 
            for k in range(0, len(spectNoisy)):
                batch_x_train[current_size] = spectNoisy[k]
                batch_y_train[current_size] = spectClear[k]
                current_size+=1
                if current_size>=batch_size:    
                    break
        return batch_x_train, batch_y_train
    
##### main logic #####

clear_files = glob.glob(r'F:\Sound\output\*wav')

x_train = []
x_train_count = 0
for i, path_clear in enumerate(clear_files):
    spectNoisy, _, _, audioNoisySR = audioToTensor(path_clear)
    x_train.append((path_clear, len(spectNoisy)))
    x_train_count += len(spectNoisy)
print("x_train_count:", x_train_count)

# Create data generator object
train_data_generator = MySequence(x_train, x_train_count, batch_size, is_noisy=True)

print('Build model...')
if os.path.exists(model_name):
    print("Load: " + model_name)
    model = load_model(model_name)
else:
    def dense_block(x, growth_rate, num_layers, dropout_rate = 0.4):
        """
        Dense block implementation with concatenation of feature maps.

        Args:
            x: Input tensor.
            growth_rate: Growth rate for the number of filters added in each layer.
            num_layers: Number of convolutional layers in the dense block.

        Returns:
            Output tensor after concatenation within the dense block.
        """
        # Store the initial input for later concatenation
        inputs = x  
    
        for _ in range(num_layers):
            # Convolutional layer
            x = tf.keras.layers.Conv1D(growth_rate, kernel_size=3, padding='same')(x)
            # Batch normalization
            x = tf.keras.layers.BatchNormalization()(x)
            # ReLU activation
            x = tf.keras.layers.Activation('relu')(x)
            # Dropout
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            # Concatenate with previous feature maps and initial input
            x = tf.keras.layers.Concatenate()([x, inputs])

        return x

    def transition_block(x):
        """
        Transition block with bottleneck layer and pooling for dimensionality reduction.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying convolution and pooling.
        """
        # Convolutional layer with fewer filters
        x = Conv1D(16, kernel_size=1, padding='same')(x)
        # Batch normalization
        x = tf.keras.layers.BatchNormalization()(x)
        # Max pooling
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        # ReLU activation
        x = tf.keras.layers.Activation('relu')(x)
        return x
    
    def get_input_shape(frame_length):
        return (int(frame_length/2+1),)

    main_input = Input(shape=(int(frame_length/2+1), 1), name='main_input')
    x = main_input
    #linear transformation of data
    x = Conv1D(1, kernel_size=1, padding='same')(x) 
    # Dense blocks with transition blocks in between
    # Growth rate for each dense block
    growth_rate = 32  
    # Number of layers per dense block
    num_layers_per_block = 3  

    
    x = dense_block(x, growth_rate, num_layers_per_block)
    x = transition_block(x)
    x = dense_block(x, growth_rate, num_layers_per_block)
    #(optional)
    #x = transition_block(x)
    #x = dense_block(x, growth_rate, num_layers_per_block)

    # Final convolutional layer
    #x = Conv1D(1, kernel_size=1, padding='same')(x)
    # Final Guided Reccurent Unit to increase dimentions
    x = GRU(257, return_sequences=False)(x)

    model = Model(inputs=main_input, outputs=x)
    tf.keras.utils.plot_model(model, to_file='model_densenet.png', show_shapes=True)

model.compile(loss='mse', metrics=['mse'], optimizer=optimiser)

print('Train...')
history = model.fit(train_data_generator, epochs=epochs, steps_per_epoch=(x_train_count // batch_size))
model.save(model_name)

metrics = history.history
plt.plot(history.epoch, metrics['mse'])
plt.legend(['mse'])
plt.savefig("learning-dense_gain.png")
plt.show()
plt.close()
