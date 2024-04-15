import os
import tensorflow as tf
import numpy as np
import random
import soundfile as sf

def is_valid_wav(filepath):
    try:
        sf.read(filepath)
        return True
    except (sf.SoundFileError, FileNotFoundError):
        return False

# Define directory paths with forward slashes (/)
speech_dir = r'F:\Sound\speech'  # Speech directory
noise_dir = r'F:\Sound\noise'    # Noise directory
output_dir = r'F:\Sound\outputnew'  # Output directory

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load speech and noise files using list comprehension
speech_files = [os.path.join(speech_dir, f) for f in os.listdir(speech_dir)
                if f.lower().endswith('.wav') or f.lower().endswith('.flac')]
noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir)
                if f.lower().endswith('.wav') or f.lower().endswith('.flac')]

# Shuffle the noise files list to ensure randomness
random.shuffle(noise_files)

for clear_filepath, random_noise_file in zip(speech_files, noise_files):
    # Read speech file
    audio_binary = tf.io.read_file(clear_filepath)
    audio_clear, audioSR_clear = tf.audio.decode_wav(audio_binary)
    audio_clear = tf.squeeze(audio_clear, axis=-1)  # Optionally squeeze if necessary

    # Extract file code from path without extension
    clear_file_code = os.path.splitext(os.path.basename(clear_filepath))[0]

    # Read noise file
    audio_binary = tf.io.read_file(random_noise_file)
    audio_noise, audioSR_noise = tf.audio.decode_wav(audio_binary)
    noise_file_code = os.path.splitext(os.path.basename(random_noise_file))[0]
    audio_noise = tf.squeeze(audio_noise, axis=-1)  # Optionally squeeze if necessary

    print("Load:", clear_filepath, clear_file_code, audioSR_clear)
    print("Load:", random_noise_file, noise_file_code, audioSR_noise)

    # Ensure audio lengths match (consider resampling if necessary)
    if len(audio_noise) != len(audio_clear):
        audio_noise = audio_noise[:len(audio_clear)]  # Slice noise to match clear audio length

    padding_amount = len(audio_clear) - len(audio_noise)
    audio_noise = tf.pad(audio_noise, paddings=[[0, padding_amount]], constant_values=0)

    # Add noise to speech
    audio_with_noise = tf.math.add(audio_clear, audio_noise)

    print(audio_clear.shape)
    print(audio_noise.shape)
    print(audio_with_noise.shape)

    # Conditionally add dimension based on input shapes
    if len(audio_with_noise.shape) == 1:
        audio_with_noise = tf.expand_dims(audio_with_noise, axis=-1)

    # Save noisy audio with proper path handling
    output_filename_noisy = os.path.join(output_dir, f"noisy_{clear_file_code}.wav")

    # Normalize audio (optional)
    # audio_with_noise /= tf.reduce_max(tf.abs(audio_with_noise))

    # Encode and save noisy audio
    wav_data_noisy = tf.audio.encode_wav(audio_with_noise, sample_rate=audioSR_clear)
    tf.io.write_file(output_filename_noisy, wav_data_noisy)
    print("Saved noisy:", output_filename_noisy)

    # Save clean audio with proper path handling
    output_filename_clean = os.path.join(output_dir, f"clean_{clear_file_code}.wav")
    tf.io.write_file(output_filename_clean, wav_data_noisy)
    print("Saved clean:", output_filename_clean)

    
