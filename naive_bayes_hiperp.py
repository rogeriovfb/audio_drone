from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from utils import plot_confusion_matrix, save_model, import_dataset

X_train, X_valid, X_test, y_train_direction, y_valid_direction, y_test_direction, y_train_fault, \
           y_valid_fault, y_test_fault, le_direction, le_fault = import_dataset()


# Função para treinar o modelo Naive Bayes
def train_naive_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

# Treinar o modelo para `maneuvering_direction`
print("Modelo para `maneuvering_direction`")
nb_model_direction = train_naive_bayes(X_train, y_train_direction)

# Treinar o modelo para `fault`
print("\nModelo para `fault`")
nb_model_fault = train_naive_bayes(X_train, y_train_fault)

# Avaliação final nos dados de teste para `maneuvering_direction`
y_pred_test_direction = nb_model_direction.predict(X_test)
print("\nAvaliação no conjunto de teste para `maneuvering_direction`")
print("Acurácia:", accuracy_score(y_test_direction, y_pred_test_direction))
print(classification_report(y_test_direction, y_pred_test_direction, target_names=le_direction.classes_))
plot_confusion_matrix(y_test_direction, y_pred_test_direction, le_direction.classes_, "Matriz de Confusão para Maneuvering Direction")

# Avaliação final nos dados de teste para `fault`
y_pred_test_fault = nb_model_fault.predict(X_test)
print("\nAvaliação no conjunto de teste para `fault`")
print("Acurácia:", accuracy_score(y_test_fault, y_pred_test_fault))
print(classification_report(y_test_fault, y_pred_test_fault, target_names=le_fault.classes_))
plot_confusion_matrix(y_test_fault, y_pred_test_fault, le_fault.classes_, "Matriz de Confusão para Fault")

save_model(nb_model_fault, nb_model_direction, 'naive_bayes')