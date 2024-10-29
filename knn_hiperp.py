import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from utils import plot_confusion_matrix

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

# Definir hiperparâmetros para o KNN
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance']
}

# Função para treinar e buscar os melhores hiperparâmetros usando o conjunto de validação
def train_best_knn(X_train, y_train, X_valid, y_valid, param_grid):
    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Melhores hiperparâmetros:", grid_search.best_params_)
    best_model.fit(np.vstack((X_train, X_valid)), np.concatenate((y_train, y_valid)))
    return best_model

# Treinar o modelo para `maneuvering_direction`
print("Modelo para `maneuvering_direction`")
best_knn_direction = train_best_knn(X_train, y_train_direction, X_valid, y_valid_direction, param_grid)

# Treinar o modelo para `fault`
print("\nModelo para `fault`")
best_knn_fault = train_best_knn(X_train, y_train_fault, X_valid, y_valid_fault, param_grid)

# Avaliação final nos dados de teste para `maneuvering_direction`
y_pred_test_direction = best_knn_direction.predict(X_test)
print("\nAvaliação no conjunto de teste para `maneuvering_direction`")
print("Acurácia:", accuracy_score(y_test_direction, y_pred_test_direction))
print(classification_report(y_test_direction, y_pred_test_direction, target_names=le_direction.classes_))
plot_confusion_matrix(y_test_direction, y_pred_test_direction, le_direction.classes_, "Matriz de Confusão para Maneuvering Direction")

# Avaliação final nos dados de teste para `fault`
y_pred_test_fault = best_knn_fault.predict(X_test)
print("\nAvaliação no conjunto de teste para `fault`")
print("Acurácia:", accuracy_score(y_test_fault, y_pred_test_fault))
print(classification_report(y_test_fault, y_pred_test_fault, target_names=le_fault.classes_))
plot_confusion_matrix(y_test_fault, y_pred_test_fault, le_fault.classes_, "Matriz de Confusão para Fault")
