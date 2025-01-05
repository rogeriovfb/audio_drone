import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from utils import save_model, plot_confusion_matrix_with_labels, plot_training_history, lr_schedule, \
    data_image_generator, save_classification_report
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Reshape


# Caminho principal do dataset de imagens
data_dir = "C:\\Github\\audio_drone\\Audio_drones_spectrograms"

train_generator, valid_generator, test_generator, fault_classes = data_image_generator(data_dir, (224, 224))


# Build LSTM Model
model = Sequential([
    Input(shape=(224, 224, 3)),  # Input shape (height, width, channels)
    Reshape((224*3, 224)),  # Reshape the input to (timesteps, features)
    LSTM(256, return_sequences=True),
    Dropout(0.3),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(fault_classes, activation='softmax')
])

print(model.summary())

# Compilar o modelo com uma taxa de aprendizado inicial menor
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks para salvar o melhor modelo e ajustar a taxa de aprendizado
early_stop_fault = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(lr_schedule)

# Treinamento do modelo com dados de validação e callback para ajuste de taxa de aprendizado
history_fault = model.fit(
    train_generator,
    epochs=30,
    validation_data=valid_generator,
    callbacks=[early_stop_fault, lr_scheduler],
    verbose=1
)

# Avaliação do modelo no conjunto de teste
test_loss_fault, test_accuracy_fault = model.evaluate(test_generator)
print("Test Accuracy (Fault):", test_accuracy_fault)

# Predições e relatório de classificação para falhas
y_true_fault = test_generator.classes
y_pred_fault = model.predict(test_generator)
y_pred_fault = np.argmax(y_pred_fault, axis=1)

# Relatório de classificação
class_labels = list(test_generator.class_indices.keys())  # Nomes das classes
print("Fault Classification Report:")
report = classification_report(y_true_fault, y_pred_fault, target_names=class_labels)
print(report)

# Salvar o classification report em um arquivo txt
save_classification_report(
    y_true_fault,
    y_pred_fault,
    class_labels,
    save_path="classification_report_fault_lstm.txt"
)

# Matriz de Confusão para falhas
conf_matrix_fault = confusion_matrix(y_true_fault, y_pred_fault)
plot_confusion_matrix_with_labels(
    conf_matrix_fault,
    class_labels,
    "Confusion Matrix for LSTM - Fault",
    save_path="confusion_matrix_fault_lstm.png"
)
save_model(model, model, "lstm")
# Plota os gráficos de treinamento e validação
plot_training_history(history_fault, save_path="training_history_fault_lstm.png")