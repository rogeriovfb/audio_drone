import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from utils import data_dir

# Define the output directory relative to the project's root folder
root_dir = Path(__file__).resolve().parent.parent  # Navigate to the root directory
output_dir = root_dir / "Audio_drones_spectrograms"
output_dir.mkdir(parents=True, exist_ok=True)


# Function to load and combine mic1 and mic2 audio as stereo
def load_stereo_audio(mic1_path, mic2_path):
    y_mic1, sr = librosa.load(mic1_path, sr=16000)  # Load mic1 with 16kHz sampling rate
    try:
        y_mic2, _ = librosa.load(mic2_path, sr=16000)  # Load mic2 with 16kHz sampling rate
        y_stereo = np.vstack((y_mic1, y_mic2))  # Combine into a stereo signal
    except FileNotFoundError:
        print(f"{mic2_path} not found. Using mono audio from mic1.")
        y_stereo = np.vstack((y_mic1, y_mic1))  # Duplicate mic1 if mic2 is missing

    return y_stereo, sr


# Function to generate stereo spectrogram and save as an image
def save_spectrogram(y_stereo, sr, save_path):
    # Create spectrograms for left and right channels
    s_left = librosa.feature.melspectrogram(y=y_stereo[0], sr=sr, n_mels=128, fmax=8000)
    s_right = librosa.feature.melspectrogram(y=y_stereo[1], sr=sr, n_mels=128, fmax=8000)

    s_left_dB = librosa.power_to_db(s_left, ref=np.max)
    s_right_dB = librosa.power_to_db(s_right, ref=np.max)

    # Configure and save the stereo spectrogram figure
    fig, axs = plt.subplots(1, 2, figsize=(2.56, 1.28), dpi=100)
    axs[0].imshow(s_left_dB, aspect='auto', origin='lower', cmap='viridis')
    axs[1].imshow(s_right_dB, aspect='auto', origin='lower', cmap='viridis')
    for ax in axs:
        ax.axis('off')
    plt.subplots_adjust(wspace=0)

    # Save and close the figure
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# Function to process a single audio file (for parallel processing)
def process_single_file(args):
    mic1_path, mic2_path, save_path = args
    try:
        y_stereo, sr = load_stereo_audio(mic1_path, mic2_path)
        save_spectrogram(y_stereo, sr, save_path)
    except Exception as e:
        print(f"Failed to process {mic1_path.name}: {e}")


# Main function to process all audio files in parallel and save stereo spectrograms
def create_spectrogram_dataset(data_dir, output_dir):
    tasks = []
    for model_type in ['A', 'B', 'C']:
        for dataset_split in ['train', 'valid', 'test']:
            audio_folder_mic1 = data_dir / model_type / dataset_split / 'mic1'
            audio_folder_mic2 = data_dir / model_type / dataset_split / 'mic2'

            output_folder = output_dir / model_type / dataset_split
            output_folder.mkdir(parents=True, exist_ok=True)

            for file_name in os.listdir(audio_folder_mic1):
                mic1_path = audio_folder_mic1 / file_name
                mic2_path = audio_folder_mic2 / file_name.replace("mic1", "mic2")
                save_path = output_folder / f"{file_name}.png"

                # Add task for parallel processing
                tasks.append((mic1_path, mic2_path, save_path))

    # Process tasks in parallel using a process pool
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_single_file, tasks), total=len(tasks), desc="Processing spectrograms"))


# Protect parallel execution
if __name__ == '__main__':
    create_spectrogram_dataset(data_dir, output_dir)
