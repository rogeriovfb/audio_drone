import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import HyperModel, RandomSearch
from sklearn.metrics import classification_report, confusion_matrix
from utils import save_model, plot_confusion_matrix_with_labels, plot_training_history, \
    save_classification_report, data_image_generator
import os

# Caminho principal do dataset de imagens
data_dir = "C:\\Github\\audio_drone\\Audio_drones_spectrograms"

# Geradores de dados com tamanho ajustado
train_generator, valid_generator, test_generator, fault_classes = data_image_generator(data_dir, (224, 224))


# Definição do espaço de busca de hiperparâmetros usando Keras Tuner
class MobileNetV2HyperModel(HyperModel):
    def build(self, hp):
        base_model_fault = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Descongelar algumas camadas
        for layer in base_model_fault.layers[-hp.Int('unfreeze_layers', 5, 10, step=5):]:
            layer.trainable = True

        x = GlobalAveragePooling2D()(base_model_fault.output)

        # Número de camadas densas
        for i in range(hp.Int('num_dense_layers', 1, 3)):
            x = Dense(hp.Int(f'dense_neurons_{i}', 256, 1024, step=256), activation='relu')(x)
            x = Dropout(hp.Choice(f'dropout_rate_{i}', [0.3, 0.5, 0.7]))(x)

        fault_output = Dense(fault_classes, activation='softmax', name='fault_output')(x)

        model = Model(inputs=base_model_fault.input, outputs=fault_output)

        # Compilação do modelo
        model.compile(
            optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-4, 5e-5, 1e-5])),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


# Configuração do Tuner
tuner = RandomSearch(
    MobileNetV2HyperModel(),
    objective='val_accuracy',
    max_trials=10,  # Número máximo de combinações de hiperparâmetros
    executions_per_trial=1,  # Número de execuções por configuração
    directory="keras_tuner_results",
    project_name="mobilenetv2_tuning"
)

# Callbacks para salvar o melhor modelo
checkpoint_callback = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Início da busca
tuner.search(
    train_generator,
    validation_data=valid_generator,
    epochs=20,
    callbacks=[checkpoint_callback]
)

# Melhor configuração de hiperparâmetros
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

# Treinar o melhor modelo com o conjunto completo de treinamento
best_model = tuner.hypermodel.build(best_hyperparameters)
history = best_model.fit(
    train_generator,
    epochs=30,
    validation_data=valid_generator,
    callbacks=[checkpoint_callback]
)

# Avaliação no conjunto de teste
test_loss_fault, test_accuracy_fault = best_model.evaluate(test_generator)
print(f"Test Accuracy (Fault): {test_accuracy_fault:.4f}")

# Predições e relatório de classificação
y_true_fault = test_generator.classes
y_pred_fault = best_model.predict(test_generator)
y_pred_fault = np.argmax(y_pred_fault, axis=1)

# Relatório de classificação
class_labels = list(test_generator.class_indices.keys())
report = classification_report(y_true_fault, y_pred_fault, target_names=class_labels)
print(report)

# Salvar o classification report em um arquivo txt
classification_report_path = "classification_report_fault_mobilenetv2_keras_tuner.txt"
save_classification_report(
    y_true_fault,
    y_pred_fault,
    class_labels,
    save_path=classification_report_path
)

# Matriz de Confusão para falhas
conf_matrix_fault = confusion_matrix(y_true_fault, y_pred_fault)
plot_confusion_matrix_with_labels(
    conf_matrix_fault,
    class_labels,
    "Confusion Matrix for MobileNetV2 - Best Fault (Keras Tuner)",
    save_path="confusion_matrix_fault_mobilenetv2_keras_tuner.png"
)

# Salvar o modelo
save_model(best_model, best_model, "mobilenetV2_best_keras_tuner")

# Plota os gráficos de treinamento e validação
plot_training_history(history, save_path="training_history_fault_mobilenetv2_keras_tuner.png")

# Salvar os hiperparâmetros em um arquivo txt
with open("best_hyperparameters_mobilenetv2.txt", "w") as f:
    f.write("Best Hyperparameters:\n")
    for key, value in best_hyperparameters.values.items():
        f.write(f"{key}: {value}\n")
    f.write(f"\nTest Accuracy: {test_accuracy_fault:.4f}\n")
