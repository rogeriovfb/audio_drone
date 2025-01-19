from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score
from utils import plot_confusion_matrix, save_model, import_dataset

X_train, X_valid, X_test, y_train_direction, y_valid_direction, y_test_direction, y_train_fault, \
           y_valid_fault, y_test_fault, le_direction, le_fault = import_dataset()


# Function to train QDA model
def train_qda(X_train, y_train):
    model = QuadraticDiscriminantAnalysis()
    model.fit(X_train, y_train)
    return model


# Train the model for `fault`
print("\nModel for `fault`")
qda_model_fault = train_qda(X_train, y_train_fault)

# Final evaluation on test data for `fault`
y_pred_test_fault = qda_model_fault.predict(X_test)
print("\nEvaluation on test set for `fault`")
print("Accuracy:", accuracy_score(y_test_fault, y_pred_test_fault))
print(classification_report(y_test_fault, y_pred_test_fault, target_names=le_fault.classes_))
plot_confusion_matrix(y_test_fault, y_pred_test_fault, le_fault.classes_, "Confusion Matrix for Fault")

save_model(qda_model_fault, 'qda')
