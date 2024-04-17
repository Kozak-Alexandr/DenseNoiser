import glob
import tensorflow as tf
import numpy as np

# Define directory paths for clean speech and noise
noise_dir = r'F:\Sound\noise'
clean_dir = r'F:\Sound\speech'

# Get list of clean and noise files
clean_files = glob.glob(f"{clean_dir}\*.wav")
noise_files = glob.glob(f"{noise_dir}\*.wav")

# Loop through each clean file
for clear_filepath in clean_files:
    # Randomly select a noise file for each clean file
    noise_file_index = np.random.randint(0, len(noise_files))
    noise_filepath = noise_files[noise_file_index]

    # Load and decode clean speech audio
    audio_binary = tf.io.read_file(clear_filepath)
    audio_clear, audioSR_clear = tf.audio.decode_wav(audio_binary)
    audio_clear = tf.squeeze(audio_clear, axis=-1)  # Remove unnecessary dimension
    clear_file_code = clear_filepath.replace(f"{clean_dir}\\", "").replace(".wav", "")  # Extract file code
    print("Load:", clear_filepath, clear_file_code, audioSR_clear)

    # Load and decode noise audio
    audio_binary = tf.io.read_file(noise_filepath)
    audio_noise, audioSR_noise = tf.audio.decode_wav(audio_binary)
    noise_dir = clean_dir.replace("speech", "noise")  # Define noise directory based on clean directory
    noise_file_code = clear_file_code.replace("speech", "noise")  # Replace "speech" with "noise" in file code
    print("Load:", noise_filepath, noise_file_code, audioSR_noise)
    audio_noise = tf.squeeze(audio_noise, axis=-1)  # Remove unnecessary dimension

    # Ensure noise is long enough for the speech, truncate if too long
    while len(audio_noise) < len(audio_clear):
        audio_noise = tf.concat([audio_noise, audio_noise], -1)
    audio_noise = audio_noise[0:len(audio_clear)]

    # Add noise to clean speech
    audio_with_noise = tf.math.add(audio_clear, audio_noise)
    audio_with_noise = tf.expand_dims(audio_with_noise, axis=-1)  # Add dimension for encoding

    # Encode audio with noise to WAV format
    audio_string = tf.audio.encode_wav(audio_with_noise, sample_rate=audioSR_clear)

    # Save noisy speech with specific naming format
    noisy_filepath = f"{noise_dir}\\{noise_file_code}.wav"
    tf.io.write_file(noisy_filepath, contents=audio_string)
    print("File saved:", noisy_filepath)

    # Save clean speech with specific naming format (optional)
    clean_filepath = f"{clean_dir}\\{clear_file_code}.wav"
    tf.io.write_file(clean_filepath, contents=tf.audio.encode_wav(tf.expand_dims(audio_clear, axis=-1), sample_rate=audioSR_clear))
    print("File saved:", clean_filepath)
