import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from utils import plot_confusion_matrix, save_model, import_dataset
from sklearn.tree import DecisionTreeClassifier

# Importar o dataset
X_train, X_valid, X_test, y_train_direction, y_valid_direction, y_test_direction, y_train_fault, \
           y_valid_fault, y_test_fault, le_direction, le_fault = import_dataset()

# Parâmetros para busca de hiperparâmetros (busca ampliada com mais opções)
param_grid = {
    'n_estimators': [50, 100, 200, 300, 500],  # Aumentando o número de estimadores
    'learning_rate': [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0, 1.2],  # Taxa de aprendizado
    'estimator__max_depth': [1, 2, 3, 4],  # Profundidade máxima da árvore de decisão
    'estimator__min_samples_split': [2, 5, 10],  # Mínimo de amostras para dividir um nó
    'estimator__min_samples_leaf': [1, 2, 4]  # Mínimo de amostras por folha
}

# Configuração do modelo base para o AdaBoost (árvore de decisão com ajuste nos hiperparâmetros)
base_estimator = DecisionTreeClassifier()

# Função para treinar e buscar os melhores hiperparâmetros usando o conjunto de validação
def train_best_adaboost(X_train, y_train, X_valid, y_valid, param_grid):
    model = AdaBoostClassifier(estimator=base_estimator)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Melhores hiperparâmetros:", grid_search.best_params_)
    best_model.fit(np.vstack((X_train, X_valid)), np.concatenate((y_train, y_valid)))
    return best_model

# Treinar o modelo para `fault`
print("\nModelo para `fault`")
best_adaboost_fault = train_best_adaboost(X_train, y_train_fault, X_valid, y_valid_fault, param_grid)

# Treinar o modelo para `maneuvering_direction`
print("Modelo para `maneuvering_direction`")
best_adaboost_direction = train_best_adaboost(X_train, y_train_direction, X_valid, y_valid_direction, param_grid)

# Salvar os modelos
save_model(best_adaboost_fault, best_adaboost_direction, 'AdaBoost')

# Avaliação final nos dados de teste para `maneuvering_direction`
y_pred_test_direction = best_adaboost_direction.predict(X_test)
print("\nAvaliação no conjunto de teste para `maneuvering_direction`")
print("Acurácia:", accuracy_score(y_test_direction, y_pred_test_direction))
print(classification_report(y_test_direction, y_pred_test_direction, target_names=le_direction.classes_))
plot_confusion_matrix(y_test_direction, y_pred_test_direction, le_direction.classes_, "Matriz de Confusão para Maneuvering Direction")

# Avaliação final nos dados de teste para `fault`
y_pred_test_fault = best_adaboost_fault.predict(X_test)
print("\nAvaliação no conjunto de teste para `fault`")
print("Acurácia:", accuracy_score(y_test_fault, y_pred_test_fault))
print(classification_report(y_test_fault, y_pred_test_fault, target_names=le_fault.classes_))
plot_confusion_matrix(y_test_fault, y_pred_test_fault, le_fault.classes_, "Matriz de Confusão para Fault")
