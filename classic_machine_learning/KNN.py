import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from utils import plot_confusion_matrix, save_model, import_dataset

# Import dataset
X_train, X_valid, X_test, y_train_direction, y_valid_direction, y_test_direction, y_train_fault, \
           y_valid_fault, y_test_fault, le_direction, le_fault = import_dataset()


# Hyperparameters for tuning
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance']
}


# Function to train and find the best hyperparameters using the validation set
def train_best_knn(X_train, y_train, X_valid, y_valid, param_grid):
    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Best hyperparameters:", grid_search.best_params_)
    best_model.fit(np.vstack((X_train, X_valid)), np.concatenate((y_train, y_valid)))
    return best_model


# Train the model for `fault`
print("\nModel for `fault`")
best_knn_fault = train_best_knn(X_train, y_train_fault, X_valid, y_valid_fault, param_grid)

# Final evaluation on test data for `fault`
y_pred_test_fault = best_knn_fault.predict(X_test)
print("\nEvaluation on test set for `fault`")
print("Accuracy:", accuracy_score(y_test_fault, y_pred_test_fault))
print(classification_report(y_test_fault, y_pred_test_fault, target_names=le_fault.classes_))
plot_confusion_matrix(y_test_fault, y_pred_test_fault, le_fault.classes_, "Confusion Matrix for Fault")

save_model(best_knn_fault, 'knn')
