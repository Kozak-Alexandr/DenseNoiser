import tensorflow as tf
import numpy as np
import os

# Load the saved model
model_path = r"F:\Code\DenseNoise"
model = tf.keras.models.load_model(model_path)

# Directory containing noisy WAV files for prediction
noisy_dir = r"F:\Sound\noisy"

# Constants for frame length
frame_length = 512

# Function to load and preprocess audio for prediction
def load_and_preprocess_audio(filepath):
    """
    Reads a WAV file, converts it to a spectrogram, and preprocesses it for prediction.

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
    frame_step = int(audioSR * 0.008)  # 16000*0.008=128
    spectrogram = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step)
    spect_image = tf.math.imag(spectrogram)
    spect_real = tf.math.real(spectrogram)
    spect_sign = tf.sign(spect_real)
    spect_real = tf.abs(spect_real)
    return spect_real, spect_image, spect_sign, audioSR

# Function to reconstruct audio from spectrogram
def spect_to_audio(spect_real, spect_sign, spect_image, audioSR):
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
    spect_real *= spect_sign
    spect_all = tf.complex(spect_real, spect_image)
    inverse_stft = tf.signal.inverse_stft(spect_all, frame_length=frame_length, frame_step=frame_step,
                                          window_fn=tf.signal.inverse_stft_window_fn(frame_step))
    return inverse_stft

# Directory to save denoised audio files
output_dir = r"F:\Sound\denoised"
os.makedirs(output_dir, exist_ok=True)

def main():
    # Process each file in the directory
    for filename in os.listdir(noisy_dir):
        if filename.endswith(".wav"):
            filepath = os.path.join(noisy_dir, filename)
            # Load and preprocess audio
            spect_real, spect_image, spect_sign, audioSR = load_and_preprocess_audio(filepath)
            # Predict denoised spectrogram
            denoised_spectrogram = model.predict(spect_real)
            # Reconstruct denoised audio from denoised spectrogram
            denoised_audio = spect_to_audio(denoised_spectrogram, spect_sign, spect_image, audioSR)
            # Reshape audio tensor to have two dimensions
            denoised_audio = tf.expand_dims(denoised_audio, axis=-1)
            # Save denoised audio
            output_filepath = os.path.join(output_dir, filename)
            wav_audio = tf.audio.encode_wav(denoised_audio, audioSR)
            tf.io.write_file(output_filepath, wav_audio)

    print("Done")

if __name__ == "__main__":
    main()
