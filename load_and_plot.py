from sklearn.metrics import classification_report, accuracy_score
from utils import plot_confusion_matrix, save_model, import_dataset
from sklearn.tree import DecisionTreeClassifier
import joblib

# Importar o dataset
X_train, X_valid, X_test, y_train_direction, y_valid_direction, y_test_direction, y_train_fault, \
           y_valid_fault, y_test_fault, le_direction, le_fault = import_dataset()

best_adaboost_fault = joblib.load('best_model_AdaBoost_fault.pkl')
best_adaboost_direction = joblib.load('best_model_AdaBoost_direction.pkl')

print(best_adaboost_fault.get_params())
print(best_adaboost_direction.get_params())

# Avaliação final nos dados de teste para `maneuvering_direction`
y_pred_test_direction = best_adaboost_direction.predict(X_test)
print("\nAvaliação no conjunto de teste para `maneuvering_direction`")
print("Acurácia:", accuracy_score(y_test_direction, y_pred_test_direction))
print(classification_report(y_test_direction, y_pred_test_direction, target_names=le_direction.classes_))
plot_confusion_matrix(y_test_direction, y_pred_test_direction, le_direction.classes_, "Confusion Matrix for Maneuvering Direction using ADABOOST")

# Avaliação final nos dados de teste para `fault`
y_pred_test_fault = best_adaboost_fault.predict(X_test)
print("\nAvaliação no conjunto de teste para `fault`")
print("Acurácia:", accuracy_score(y_test_fault, y_pred_test_fault))
print(classification_report(y_test_fault, y_pred_test_fault, target_names=le_fault.classes_))
plot_confusion_matrix(y_test_fault, y_pred_test_fault, le_fault.classes_, "Confusion Matrix for Fault using ADABOOST")
