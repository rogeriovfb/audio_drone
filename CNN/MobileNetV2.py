import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import HyperModel, RandomSearch
from sklearn.metrics import classification_report, confusion_matrix
from utils import (
    save_model,
    plot_confusion_matrix_with_labels,
    plot_training_history,
    save_classification_report,
    data_image_generator
)

# Load dataset path from project root
data_dir = os.path.join(os.path.dirname(__file__), "../Audio_drones_spectrograms")

# Generate data loaders with adjusted size
train_generator, valid_generator, test_generator, fault_classes = data_image_generator(data_dir, (224, 224))


# Define hyperparameter search space using Keras Tuner
class MobileNetV2HyperModel(HyperModel):
    def build(self, hp):
        base_model_fault = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Unfreeze specific layers
        for layer in base_model_fault.layers[-hp.Int('unfreeze_layers', 5, 10, step=5):]:
            layer.trainable = True

        x = GlobalAveragePooling2D()(base_model_fault.output)

        # Add dense layers based on hyperparameters
        for i in range(hp.Int('num_dense_layers', 1, 3)):
            x = Dense(hp.Int(f'dense_neurons_{i}', 256, 1024, step=256), activation='relu')(x)
            x = Dropout(hp.Choice(f'dropout_rate_{i}', [0.3, 0.5, 0.7]))(x)

        fault_output = Dense(fault_classes, activation='softmax', name='fault_output')(x)

        model = Model(inputs=base_model_fault.input, outputs=fault_output)

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-4, 5e-5, 1e-5])),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


# Configure Keras Tuner
tuner = RandomSearch(
    MobileNetV2HyperModel(),
    objective='val_accuracy',
    max_trials=15,
    executions_per_trial=1,
    directory="keras_tuner_results",
    project_name="mobilenetv2_tuning"
)

# Configure callbacks for training
checkpoint_callback = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Start hyperparameter search
tuner.search(
    train_generator,
    validation_data=valid_generator,
    epochs=20,
    callbacks=[checkpoint_callback]
)

# Retrieve best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

# Train the best model with the complete training dataset
best_model = tuner.hypermodel.build(best_hyperparameters)
history = best_model.fit(
    train_generator,
    epochs=30,
    validation_data=valid_generator,
    callbacks=[checkpoint_callback]
)

# Evaluate the model on the test set
test_loss_fault, test_accuracy_fault = best_model.evaluate(test_generator)
print(f"Test Accuracy (Fault): {test_accuracy_fault:.4f}")

# Generate predictions and classification report
y_true_fault = test_generator.classes
y_pred_fault = best_model.predict(test_generator)
y_pred_fault = np.argmax(y_pred_fault, axis=1)

class_labels = list(test_generator.class_indices.keys())
report = classification_report(y_true_fault, y_pred_fault, target_names=class_labels)
print(report)

# Save the classification report
classification_report_path = "classification_report_fault_mobilenetv2_keras_tuner.txt"
save_classification_report(
    y_true_fault,
    y_pred_fault,
    class_labels,
    save_path=classification_report_path
)

# Generate and save the confusion matrix
conf_matrix_fault = confusion_matrix(y_true_fault, y_pred_fault)
plot_confusion_matrix_with_labels(
    conf_matrix_fault,
    class_labels,
    "Confusion Matrix for MobileNetV2 - Best Fault (Keras Tuner)",
    save_path="confusion_matrix_fault_mobilenetv2_keras_tuner.png"
)

# Save the best model
save_model(best_model, "mobilenetV2_best_keras_tuner")

# Plot and save training history
plot_training_history(history, save_path="training_history_fault_mobilenetv2_keras_tuner.png")

# Save the best hyperparameters to a file
with open("best_hyperparameters_mobilenetv2.txt", "w") as f:
    f.write("Best Hyperparameters:\n")
    for key, value in best_hyperparameters.values.items():
        f.write(f"{key}: {value}\n")
    f.write(f"\nTest Accuracy: {test_accuracy_fault:.4f}\n")
