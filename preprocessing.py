import os
import librosa
import pandas as pd
from tqdm import tqdm

# Atualizar a função `extract_features` para receber `y` e `sr` diretamente
def extract_features(y, sr):
    features = {}

    # Coeficientes MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(mfcc.shape[0]):
        features[f'mfcc_{i}'] = mfcc[i].mean()

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    features['spectral_centroid'] = spectral_centroid

    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    features['spectral_bandwidth'] = spectral_bandwidth

    # Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean()
    features['spectral_contrast'] = spectral_contrast

    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85).mean()
    features['spectral_rolloff'] = spectral_rolloff

    # Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
    features['zero_crossing_rate'] = zero_crossing_rate

    # Root Mean Square Energy (RMSE)
    rmse = librosa.feature.rms(y=y).mean()
    features['rmse'] = rmse

    return features

# Carregar os dados e extrair as features
def process_audio_data(data_dir):
    rows = []

    # Iterar pelas pastas e arquivos
    for model in ['A', 'B', 'C']:
        for split in ['train', 'test', 'valid']:
            mic1_dir = os.path.join(data_dir, model, split, 'mic1')
            mic2_dir = os.path.join(data_dir, model, split, 'mic2')

            for filename in tqdm(os.listdir(mic1_dir), desc=f'Processing {model}/{split}'):
                file_path_mic1 = os.path.join(mic1_dir, filename)
                file_path_mic2 = os.path.join(mic2_dir, filename.replace("mic1", "mic2"))

                try:
                    # Tentar carregar áudio estéreo (mic1 e mic2 juntos)
                    y1, sr1 = librosa.load(file_path_mic1, sr=None)
                    y2, sr2 = librosa.load(file_path_mic2, sr=None)

                    # Verificar se as taxas de amostragem são iguais antes de combinar
                    if sr1 == sr2:
                        y = (y1 + y2) / 2  # Combina os sinais dos dois microfones

                        # Extrair características estéreo combinadas
                        features = extract_features(y, sr1)
                        features['mic1'] = filename
                        features['mic2'] = filename.replace("mic1", "mic2")
                    else:
                        print(f"Taxas de amostragem diferentes para {filename}, ignorando arquivo.")

                except FileNotFoundError:
                    print(f"Arquivo estéreo não encontrado para {filename}; processando como mono.")

                    # Extrair características apenas para mic1
                    y1, sr1 = librosa.load(file_path_mic1, sr=None)
                    features = extract_features(y1, sr1)
                    features['mic1'] = filename
                    features['mic2'] = None  # Mic2 não disponível

                # Analisar nome do arquivo para rótulos adicionais
                parts = filename.split('_')
                features['model_type'] = parts[0]
                features['maneuvering_direction'] = parts[1]
                features['fault'] = parts[2]
                features['background'] = parts[5]
                features['snr'] = float(parts[-1].split('=')[1].replace('.wav', ''))

                # Split do conjunto de dados
                features['dataset_split'] = split
                rows.append(features)

    # Salvar em DataFrame e exportar
    df = pd.DataFrame(rows)
    df.to_csv("audio_drone_features_extended.csv", index=False)
    print("Extração de features concluída e salva em 'audio_drone_features_extended.csv'.")

# Defina o caminho dos dados e inicie o processamento
data_dir = "D:\\Audio_drones"
process_audio_data(data_dir)