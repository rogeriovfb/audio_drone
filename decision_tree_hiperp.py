import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from utils import plot_confusion_matrix, save_model, import_dataset

X_train, X_valid, X_test, y_train_direction, y_valid_direction, y_test_direction, y_train_fault, \
           y_valid_fault, y_test_fault, le_direction, le_fault = import_dataset()

# Hiperparâmetros para ajuste
param_grid = {
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}

# Função para treinar e buscar os melhores hiperparâmetros usando o conjunto de validação
def train_best_model(X_train, y_train, X_valid, y_valid, param_grid):
    model = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Melhores hiperparâmetros:", grid_search.best_params_)
    best_model.fit(np.vstack((X_train, X_valid)), np.concatenate((y_train, y_valid)))
    return best_model

# Treinar o modelo para `maneuvering_direction`
print("Modelo para `maneuvering_direction`")
best_model_direction = train_best_model(X_train, y_train_direction, X_valid, y_valid_direction, param_grid)

# Treinar o modelo para `fault`
print("\nModelo para `fault`")
best_model_fault = train_best_model(X_train, y_train_fault, X_valid, y_valid_fault, param_grid)

# Avaliação final nos dados de teste para `maneuvering_direction`
y_pred_test_direction = best_model_direction.predict(X_test)
print("\nAvaliação no conjunto de teste para `maneuvering_direction`")
print("Acurácia:", accuracy_score(y_test_direction, y_pred_test_direction))
print(classification_report(y_test_direction, y_pred_test_direction, target_names=le_direction.classes_))
plot_confusion_matrix(y_test_direction, y_pred_test_direction, le_direction.classes_, "Matriz de Confusão para Maneuvering Direction")

# Avaliação final nos dados de teste para `fault`
y_pred_test_fault = best_model_fault.predict(X_test)
print("\nAvaliação no conjunto de teste para `fault`")
print("Acurácia:", accuracy_score(y_test_fault, y_pred_test_fault))
print(classification_report(y_test_fault, y_pred_test_fault, target_names=le_fault.classes_))
plot_confusion_matrix(y_test_fault, y_pred_test_fault, le_fault.classes_, "Matriz de Confusão para Fault")

save_model(best_model_fault, best_model_direction, 'decision_tree')