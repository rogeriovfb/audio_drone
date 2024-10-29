import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset bruto
data = pd.read_csv("audio_drone_features_extended.csv")

# Configurar estilo de visualização do seaborn
sns.set(style="whitegrid")

# Criar uma figura com subplots para os gráficos restantes
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Análise Exploratória do Dataset de Áudio de Drones", fontsize=16)

# 1. Quantidade de dados para cada classe de direção de manobra
sns.countplot(data=data, x='maneuvering_direction', palette="viridis", order=data['maneuvering_direction'].value_counts().index, ax=axes[0, 0])
axes[0, 0].set_title("Distribuição das Classes de Direção de Manobra")
axes[0, 0].set_xlabel("Direção de Manobra")
axes[0, 0].set_ylabel("Quantidade de Amostras")

# 2. Quantidade de dados para cada condição de falha
sns.countplot(data=data, x='fault', palette="viridis", order=data['fault'].value_counts().index, ax=axes[0, 1])
axes[0, 1].set_title("Distribuição das Classes de Condição de Falha")
axes[0, 1].set_xlabel("Condição de Falha")
axes[0, 1].set_ylabel("Quantidade de Amostras")
axes[0, 1].tick_params(axis='x', rotation=45)

# 4. Histograma dos níveis de ruído (SNR) nas amostras
sns.histplot(data=data, x='snr', bins=20, kde=True, color="skyblue", ax=axes[1, 0])
axes[1, 0].set_title("Distribuição dos Níveis de Ruído (SNR)")
axes[1, 0].set_xlabel("SNR (dB)")
axes[1, 0].set_ylabel("Frequência")

# 5. Distribuição de tipos de drones
sns.countplot(data=data, x='model_type', palette="magma", order=data['model_type'].value_counts().index, ax=axes[1, 1])
axes[1, 1].set_title("Distribuição por Tipo de Drone")
axes[1, 1].set_xlabel("Tipo de Drone")
axes[1, 1].set_ylabel("Quantidade de Amostras")

# Ajustar layout para evitar sobreposição de títulos e rótulos
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
