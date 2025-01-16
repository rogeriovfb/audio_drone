import os
import librosa
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from utils import data_dir


# Function to extract features directly from audio data
def extract_features(y, sr):
    features = {}

    # MFCC coefficients
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(mfcc.shape[0]):
        features[f'mfcc_{i}'] = mfcc[i].mean()

    # Spectral Centroid
    features['spectral_centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr).mean()

    # Spectral Bandwidth
    features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()

    # Spectral Contrast
    features['spectral_contrast'] = librosa.feature.spectral_contrast(y=y, sr=sr).mean()

    # Spectral Rolloff
    features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85).mean()

    # Zero Crossing Rate
    features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(y).mean()

    # Root Mean Square Energy (RMSE)
    features['rmse'] = librosa.feature.rms(y=y).mean()

    return features


# Process audio data and extract features
def process_audio_data(dir):
    rows = []

    # Iterate through folders and files
    for model in ['A', 'B', 'C']:
        for split in ['train', 'test', 'valid']:
            mic1_dir = os.path.join(dir, model, split, 'mic1')
            mic2_dir = os.path.join(dir, model, split, 'mic2')

            for filename in tqdm(os.listdir(mic1_dir), desc=f'Processing {model}/{split}'):
                file_path_mic1 = os.path.join(mic1_dir, filename)
                file_path_mic2 = os.path.join(mic2_dir, filename.replace("mic1", "mic2"))

                try:
                    # Attempt to load stereo audio (mic1 and mic2 combined)
                    y1, sr1 = librosa.load(file_path_mic1, sr=None)
                    y2, sr2 = librosa.load(file_path_mic2, sr=None)

                    # Verify sampling rates match before combining
                    if sr1 == sr2:
                        y = (y1 + y2) / 2  # Combine mic1 and mic2 signals
                        features = extract_features(y, sr1)
                        features['mic1'] = filename
                        features['mic2'] = filename.replace("mic1", "mic2")
                    else:
                        print(f"Sampling rates differ for {filename}, skipping file.")

                except FileNotFoundError:
                    print(f"Stereo file not found for {filename}; processing as mono.")
                    y1, sr1 = librosa.load(file_path_mic1, sr=None)
                    features = extract_features(y1, sr1)
                    features['mic1'] = filename
                    features['mic2'] = None  # Mic2 unavailable

                # Parse filename for additional labels
                parts = filename.split('_')
                features['model_type'] = parts[0]
                features['maneuvering_direction'] = parts[1]
                features['fault'] = parts[2]
                features['background'] = parts[5]
                features['snr'] = float(parts[-1].split('=')[1].replace('.wav', ''))

                # Dataset split
                features['dataset_split'] = split
                rows.append(features)

    # Save DataFrame to root directory
    root_dir = Path(__file__).resolve().parent.parent  # Navigate to the project's root directory
    output_path = root_dir / "audio_drone_features_extended.csv"
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Feature extraction completed and saved to '{output_path}'.")


# Run the feature extraction
process_audio_data(data_dir)
