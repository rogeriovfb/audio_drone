import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

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