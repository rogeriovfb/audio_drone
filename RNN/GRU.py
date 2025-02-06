import os
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import HyperModel, RandomSearch
from sklearn.metrics import classification_report, confusion_matrix
from utils import save_model, plot_confusion_matrix_with_labels, plot_training_history, \
    save_classification_report, data_image_generator


# Load dataset path from project root
data_dir = os.path.join(os.path.dirname(__file__), "../Audio_drones_spectrograms")

# Data generators
train_generator, valid_generator, test_generator, fault_classes = data_image_generator(data_dir, (224, 224))


class GRUHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Input(shape=(224, 224, 3))),  # Input shape (height, width, channels)
        model.add(Reshape((224 * 3, 224))),  # Reshape the input to (timesteps, features)

        # GRU layers
        for i in range(hp.Int('num_gru_layers', 1, 3)):
            model.add(GRU(
                units=hp.Int(f'gru_units_{i}', min_value=32, max_value=256, step=32),
                return_sequences=(i < hp.Int('num_gru_layers', 1, 3) - 1),
                # Return sequences for all but last GRU layer
                activation='tanh'
            ))
            model.add(Dropout(hp.Float(f'dropout_rate_{i}', 0.2, 0.5, step=0.1)))

        # Fully connected layers
        model.add(Dense(
            units=hp.Int('dense_units', 64, 512, step=64),
            activation='relu'))
        model.add(Dropout(hp.Float('dense_dropout_rate', 0.2, 0.5, step=0.1)))

        # Output layer
        model.add(Dense(fault_classes, activation='softmax'))

        # Compile the model
        model.compile(
            optimizer=Adam(hp.Choice('learning_rate', [1e-4, 5e-5, 1e-5])),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model


# Hyperparameter tuner setup
tuner = RandomSearch(
    GRUHyperModel(),
    objective='val_accuracy',
    max_trials=15,  # Number of hyperparameter combinations to try
    executions_per_trial=1,  # Number of executions per combination
    directory="keras_tuner_results",
    project_name="gru_tuning"
)

# Early stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Perform hyperparameter search
tuner.search(
    train_generator,
    validation_data=valid_generator,
    epochs=20,
    callbacks=[early_stopping],
    verbose=1
)

# Get the best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build and train the model with the best hyperparameters
best_model = tuner.hypermodel.build(best_hyperparameters)
history = best_model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=30,
    callbacks=[early_stopping]
)

# Evaluate on the test set
test_loss, test_accuracy = best_model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Generate classification report
y_true = test_generator.classes
y_pred = best_model.predict(test_generator)
y_pred_classes = tf.argmax(y_pred, axis=1).numpy()

class_labels = list(test_generator.class_indices.keys())
report = classification_report(y_true, y_pred_classes, target_names=class_labels)
print(report)

# Save the classification report
save_classification_report(y_true, y_pred_classes, class_labels, "classification_report_gru_tuning.txt")

# Save the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plot_confusion_matrix_with_labels(
    conf_matrix,
    class_labels,
    "Confusion Matrix for GRU - Best Model",
    save_path="confusion_matrix_gru_tuning.png"
)

# Save the model
save_model(best_model, "gru_best_model")

# Save the training history
plot_training_history(history, save_path="training_history_gru_tuning.png")

# Save best hyperparameters
with open("best_hyperparameters_gru.txt", "w") as f:
    f.write("Best Hyperparameters:\n")
    for key, value in best_hyperparameters.values.items():
        f.write(f"{key}: {value}\n")
    f.write(f"\nTest Accuracy: {test_accuracy:.4f}\n")
