import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Diretório dos dados brutos de áudio
data_dir = Path("D:/Audio_drones")

# Diretório de saída para salvar os espectrogramas
output_dir = Path("D:/Audio_drones_spectrograms")
output_dir.mkdir(exist_ok=True, parents=True)


# Função para carregar e combinar áudios de mic1 e mic2 em estéreo
def load_stereo_audio(mic1_path, mic2_path):
    y_mic1, sr = librosa.load(mic1_path, sr=16000)  # Carregar mic1 com amostragem de 16kHz
    try:
        y_mic2, _ = librosa.load(mic2_path, sr=16000)  # Carregar mic2 com amostragem de 16kHz
        y_stereo = np.vstack((y_mic1, y_mic2))  # Combinar em um sinal estéreo
    except FileNotFoundError:
        print(f"{mic2_path} not found. Using mono audio from mic1.")
        y_stereo = np.vstack((y_mic1, y_mic1))  # Caso mic2 não exista, duplicar mic1

    return y_stereo, sr


# Função para gerar espectrograma estéreo e salvar como imagem
def save_spectrogram(y_stereo, sr, save_path):
    # Criar o espectrograma para os canais esquerdo e direito
    S_left = librosa.feature.melspectrogram(y=y_stereo[0], sr=sr, n_mels=128, fmax=8000)
    S_right = librosa.feature.melspectrogram(y=y_stereo[1], sr=sr, n_mels=128, fmax=8000)

    S_left_dB = librosa.power_to_db(S_left, ref=np.max)
    S_right_dB = librosa.power_to_db(S_right, ref=np.max)

    # Configurar e salvar a figura de espectrograma estéreo
    fig, axs = plt.subplots(1, 2, figsize=(2.56, 1.28), dpi=100)
    axs[0].imshow(S_left_dB, aspect='auto', origin='lower', cmap='viridis')
    axs[1].imshow(S_right_dB, aspect='auto', origin='lower', cmap='viridis')
    for ax in axs:
        ax.axis('off')
    plt.subplots_adjust(wspace=0)

    # Salvar e fechar a figura
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# Função para processar um único arquivo de áudio (ajuda no processamento paralelo)
def process_single_file(args):
    mic1_path, mic2_path, save_path = args
    try:
        y_stereo, sr = load_stereo_audio(mic1_path, mic2_path)
        save_spectrogram(y_stereo, sr, save_path)
    except Exception as e:
        print(f"Failed to process {mic1_path.name}: {e}")


# Função principal para processar todos os áudios em paralelo e salvar espectrogramas estéreo
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

                # Adicionar tarefa para processamento paralelo
                tasks.append((mic1_path, mic2_path, save_path))

    # Processar as tarefas em paralelo com um pool de processos
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_single_file, tasks), total=len(tasks), desc="Processing spectrograms"))


# Proteger a execução do código paralelo
if __name__ == '__main__':
    create_spectrogram_dataset(data_dir, output_dir)
