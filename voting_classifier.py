import joblib
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from utils import import_dataset, plot_confusion_matrix, save_model

# Importar o dataset
X_train, X_valid, X_test, y_train_direction, y_valid_direction, y_test_direction, y_train_fault, \
           y_valid_fault, y_test_fault, le_direction, le_fault = import_dataset()

# Carregar modelos salvos para direção de manobra
ada_direction = joblib.load("best_model_AdaBoost_direction.pkl")
dt_direction = joblib.load("best_model_decision_tree_direction.pkl")
knn_direction = joblib.load("best_model_knn_direction.pkl")
lr_direction = joblib.load("best_model_logistic_regression_direction.pkl")
nb_direction = joblib.load("best_model_naive_bayes_direction.pkl")
qda_direction = joblib.load("best_model_qda_direction.pkl")
rf_direction = joblib.load("best_model_random_forest_direction.pkl")
svm_direction = joblib.load("best_model_SVM_direction.pkl")
xgb_direction = joblib.load("best_model_XGBoost_direction.pkl")

# Criar o VotingClassifier para direção de manobra
voting_clf_direction = VotingClassifier(estimators=[
    ('AdaBoost', ada_direction),
    ('DecisionTree', dt_direction),
    ('KNN', knn_direction),
    ('LogisticRegression', lr_direction),
    ('NaiveBayes', nb_direction),
    ('QDA', qda_direction),
    ('RandomForest', rf_direction),
    ('SVM', svm_direction),
    ('XGBoost', xgb_direction)
], voting='soft')

# Treinar o VotingClassifier para direção de manobra no conjunto combinado
voting_clf_direction.fit(X_train, y_train_direction)

# Carregar modelos salvos para tipo e localização de falha
ada_fault = joblib.load("best_model_AdaBoost_fault.pkl")
dt_fault = joblib.load("best_model_decision_tree_fault.pkl")
knn_fault = joblib.load("best_model_knn_fault.pkl")
lr_fault = joblib.load("best_model_logistic_regression_fault.pkl")
nb_fault = joblib.load("best_model_naive_bayes_fault.pkl")
qda_fault = joblib.load("best_model_qda_fault.pkl")
rf_fault = joblib.load("best_model_random_forest_fault.pkl")
svm_fault = joblib.load("best_model_SVM_fault.pkl")
xgb_fault = joblib.load("best_model_XGBoost_fault.pkl")

# Criar o VotingClassifier para tipo e localização de falha
voting_clf_fault = VotingClassifier(estimators=[
    ('AdaBoost', ada_fault),
    ('DecisionTree', dt_fault),
    ('KNN', knn_fault),
    ('LogisticRegression', lr_fault),
    ('NaiveBayes', nb_fault),
    ('QDA', qda_fault),
    ('RandomForest', rf_fault),
    ('SVM', svm_fault),
    ('XGBoost', xgb_fault)
], voting='soft')

# Treinar o VotingClassifier para tipo e localização de falha no conjunto combinado
voting_clf_fault.fit(X_train, y_train_fault)

save_model(voting_clf_fault, voting_clf_direction, 'voting_clf')

# Previsão e avaliação para tipo e localização de falha
y_pred_test_fault = voting_clf_fault.predict(X_test)
print("\nVoting Classifier - Fault Classification")
print("Accuracy:", accuracy_score(y_test_fault, y_pred_test_fault))
print(classification_report(y_test_fault, y_pred_test_fault, target_names=le_fault.classes_))
plot_confusion_matrix(y_test_fault, y_pred_test_fault, le_fault.classes_, "Confusion Matrix - Fault Classification")


# Previsão e avaliação para direção de manobra
y_pred_test_direction = voting_clf_direction.predict(X_test)
print("\nVoting Classifier - Direction of Maneuvering")
print("Accuracy:", accuracy_score(y_test_direction, y_pred_test_direction))
print(classification_report(y_test_direction, y_pred_test_direction, target_names=le_direction.classes_))
plot_confusion_matrix(y_test_direction, y_pred_test_direction, le_direction.classes_, "Confusion Matrix - Direction of Maneuvering")