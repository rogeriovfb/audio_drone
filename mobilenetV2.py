
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from utils import save_model, plot_confusion_matrix_with_labels, plot_training_history, lr_schedule, \
    data_image_generator, save_classification_report

# Caminho principal do dataset de imagens
data_dir = "C:\\Github\\audio_drone\\Audio_drones_spectrograms"


train_generator, valid_generator, test_generator, fault_classes = data_image_generator(data_dir, (224, 224))

# Configuração do modelo MobileNetV2 com camadas adicionais e Dropout
base_model_fault = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model_fault.trainable = False  # Congelar as camadas do modelo base

x = GlobalAveragePooling2D()(base_model_fault.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
fault_output = Dense(fault_classes, activation='softmax', name='fault_output')(x)

model_fault = Model(inputs=base_model_fault.input, outputs=fault_output)
print(model_fault.summary())

# Compilar o modelo com uma taxa de aprendizado inicial menor
model_fault.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks para salvar o melhor modelo e ajustar a taxa de aprendizado
checkpoint_fault = ModelCheckpoint("best_mobilenetv2_fault.keras", monitor='val_loss', save_best_only=True, mode='min')
early_stop_fault = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(lr_schedule)

# Treinamento do modelo com dados de validação e callback para ajuste de taxa de aprendizado
history_fault = model_fault.fit(
    train_generator,
    epochs=30,
    validation_data=valid_generator,
    callbacks=[checkpoint_fault, early_stop_fault, lr_scheduler],
    verbose=1
)

# Avaliação do modelo no conjunto de teste
test_loss_fault, test_accuracy_fault = model_fault.evaluate(test_generator)
print("Test Accuracy (Fault):", test_accuracy_fault)

# Predições e relatório de classificação para falhas
y_true_fault = test_generator.classes
y_pred_fault = model_fault.predict(test_generator)
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
    save_path="classification_report_fault_mobilenetv2.txt"
)

# Matriz de Confusão para falhas
conf_matrix_fault = confusion_matrix(y_true_fault, y_pred_fault)
plot_confusion_matrix_with_labels(
    conf_matrix_fault,
    class_labels,
    "Confusion Matrix for MobileNetV2 - Fault",
    save_path="confusion_matrix_fault_mobilenetv2.png"
)
save_model(model_fault, model_fault, "mobilenetV2")
# Plota os gráficos de treinamento e validação
plot_training_history(history_fault, save_path="training_history_fault_mobilenetv2.png")

