import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import re
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

# Diretório dos dados brutos de áudio
data_dir = Path("D:/Audio_drones")

# Função para exibir a matriz de confusão
def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()


# Função para salvar os modelos
def save_model(model_fault, model_direction, algoritmo: str):
    joblib.dump(model_direction, 'best_model_' + algoritmo + '_direction.pkl')
    joblib.dump(model_fault, 'best_model_' + algoritmo + '_fault.pkl')


def import_dataset():
    # Carregar o dataset
    data = pd.read_csv("audio_drone_features_extended.csv")

    # Dividir o dataset em treinamento, validação e teste
    train_data = data[data['dataset_split'] == 'train']
    valid_data = data[data['dataset_split'] == 'valid']
    test_data = data[data['dataset_split'] == 'test']

    # Codificação do `model_type` e demais features
    le_model = LabelEncoder()
    train_data['model_type_encoded'] = le_model.fit_transform(train_data['model_type'])
    valid_data['model_type_encoded'] = le_model.transform(valid_data['model_type'])
    test_data['model_type_encoded'] = le_model.transform(test_data['model_type'])

    # Definir características (X) incluindo `model_type` e as features MFCCs
    features = [f'mfcc_{i}' for i in range(13)] + ['model_type_encoded']
    features.extend(['spectral_centroid', 'spectral_bandwidth', 'spectral_contrast', 'spectral_rolloff',
                     'zero_crossing_rate', 'rmse'])

    X_train = train_data[features]
    X_valid = valid_data[features]
    X_test = test_data[features]

    # Normalização das features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    # Codificação de rótulos para `maneuvering_direction` e `fault`
    le_direction = LabelEncoder()
    y_train_direction = le_direction.fit_transform(train_data['maneuvering_direction'])
    y_valid_direction = le_direction.transform(valid_data['maneuvering_direction'])
    y_test_direction = le_direction.transform(test_data['maneuvering_direction'])

    le_fault = LabelEncoder()
    y_train_fault = le_fault.fit_transform(train_data['fault'])
    y_valid_fault = le_fault.transform(valid_data['fault'])
    y_test_fault = le_fault.transform(test_data['fault'])

    return X_train, X_valid, X_test, y_train_direction, y_valid_direction, y_test_direction, y_train_fault, \
           y_valid_fault, y_test_fault, le_direction, le_fault


def plot_confusion_matrix_with_labels(cm, class_names, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.savefig(save_path)  # Salva o gráfico
    plt.close()  # Fecha para evitar sobrecarga de memória


# Função para plotar os gráficos de treinamento e validação
def plot_training_history(history, save_path):
    plt.figure(figsize=(10, 5))
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')

    plt.tight_layout()
    plt.savefig(save_path)  # Salva o gráfico
    plt.close()  # Fecha para evitar sobrecarga de memória


# Função de agendamento da taxa de aprendizado
def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    elif epoch < 20:
        return lr * 0.9
    else:
        return lr * 0.5


def data_image_generator(data_dir, size):
    # Função para extrair rótulo de falha dos nomes dos arquivos
    def extract_fault_label(filename):
        pattern = r"_(N|MF[1-4]|PC[1-4])_"
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
        return None

    # Criar geradores de dados para treino, validação e teste
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    valid_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Definir um DataFrame com caminhos de arquivos e rótulos extraídos
    data_fault = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".png"):
                fault = extract_fault_label(file)
                if fault:
                    folder_name = os.path.basename(root)
                    data_fault.append({
                        "filepath": os.path.join(root, file),
                        "fault": fault,
                        "set": folder_name  # Conjunto (train, test, valid)
                    })

    df_fault = pd.DataFrame(data_fault)
    df_fault['fault'] = df_fault['fault'].astype('category')
    fault_classes = len(df_fault['fault'].unique())

    # Separar os dados em DataFrames para cada conjunto
    train_df_fault = df_fault[df_fault['set'] == 'train']
    valid_df_fault = df_fault[df_fault['set'] == 'valid']
    test_df_fault = df_fault[df_fault['set'] == 'test']

    # Configurar os geradores de dados usando ImageDataGenerator
    train_generator = train_datagen.flow_from_dataframe(
        train_df_fault,
        x_col="filepath",
        y_col="fault",
        target_size=size,
        class_mode="categorical",
        batch_size=32,
        shuffle=True
    )

    valid_generator = valid_datagen.flow_from_dataframe(
        valid_df_fault,
        x_col="filepath",
        y_col="fault",
        target_size=size,
        class_mode="categorical",
        batch_size=32,
        shuffle=False
    )

    test_generator = test_datagen.flow_from_dataframe(
        test_df_fault,
        x_col="filepath",
        y_col="fault",
        target_size=size,
        class_mode="categorical",
        batch_size=32,
        shuffle=False
    )

    return train_generator, valid_generator, test_generator, fault_classes

def save_classification_report(y_true, y_pred, class_labels, save_path):
    report = classification_report(y_true, y_pred, target_names=class_labels)
    with open(save_path, "w") as f:
        f.write("Classification Report\n")
        f.write("=====================\n")
        f.write(report)