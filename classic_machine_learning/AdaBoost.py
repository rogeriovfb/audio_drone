import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from utils import plot_confusion_matrix, save_model, import_dataset
from sklearn.tree import DecisionTreeClassifier

# Import the dataset
X_train, X_valid, X_test, y_train_direction, y_valid_direction, y_test_direction, y_train_fault, \
           y_valid_fault, y_test_fault, le_direction, le_fault = import_dataset()

# Parameters for hyperparameter tuning (expanded search with more options)
param_grid = {
    'n_estimators': [50, 100, 200, 300, 500],  # Increasing the number of estimators
    'learning_rate': [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0, 1.2],  # Learning rate
    'estimator__max_depth': [1, 2, 3, 4],  # Maximum depth of the decision tree
    'estimator__min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'estimator__min_samples_leaf': [1, 2, 4]  # Minimum samples required at a leaf node
}

# Configure the base model for AdaBoost (decision tree with tuned hyperparameters)
base_estimator = DecisionTreeClassifier()


# Function to train and find the best hyperparameters using the validation set
def train_best_adaboost(X_train, y_train, X_valid, y_valid, param_grid):
    model = AdaBoostClassifier(estimator=base_estimator, algorithm='SAMME')
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Best hyperparameters:", grid_search.best_params_)
    best_model.fit(np.vstack((X_train, X_valid)), np.concatenate((y_train, y_valid)))
    return best_model


# Train the model for `fault`
print("\nModel for `fault`")
best_adaboost_fault = train_best_adaboost(X_train, y_train_fault, X_valid, y_valid_fault, param_grid)

# Save the models
save_model(best_adaboost_fault, 'AdaBoost')

# Final evaluation on test data for `fault`
y_pred_test_fault = best_adaboost_fault.predict(X_test)
print("\nEvaluation on test set for `fault`")
print("Accuracy:", accuracy_score(y_test_fault, y_pred_test_fault))
print(classification_report(y_test_fault, y_pred_test_fault, target_names=le_fault.classes_))
plot_confusion_matrix(y_test_fault, y_pred_test_fault, le_fault.classes_, "Confusion Matrix for Fault")

