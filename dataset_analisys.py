import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the raw dataset
data = pd.read_csv("audio_drone_features_extended.csv")

# Set seaborn visualization style
sns.set(style="whitegrid")

# Create a figure with subplots for the remaining charts
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Exploratory Analysis of the Drone Audio Dataset", fontsize=16)

# 1. Data count for each maneuvering direction class
sns.countplot(data=data, x='maneuvering_direction', palette="viridis", order=data['maneuvering_direction'].value_counts().index, ax=axes[0, 0])
axes[0, 0].set_title("Distribution of Maneuvering Direction Classes")
axes[0, 0].set_xlabel("Maneuvering Direction")
axes[0, 0].set_ylabel("Number of Samples")

# 2. Data count for each fault condition class
sns.countplot(data=data, x='fault', palette="viridis", order=data['fault'].value_counts().index, ax=axes[0, 1])
axes[0, 1].set_title("Distribution of Fault Condition Classes")
axes[0, 1].set_xlabel("Fault Condition")
axes[0, 1].set_ylabel("Number of Samples")
axes[0, 1].tick_params(axis='x', rotation=45)

# 4. Histogram of noise levels (SNR) in samples
sns.histplot(data=data, x='snr', bins=20, kde=True, color="skyblue", ax=axes[1, 0])
axes[1, 0].set_title("Distribution of Noise Levels (SNR)")
axes[1, 0].set_xlabel("SNR (dB)")
axes[1, 0].set_ylabel("Frequency")

# 5. Distribution by drone types
sns.countplot(data=data, x='model_type', palette="magma", order=data['model_type'].value_counts().index, ax=axes[1, 1])
axes[1, 1].set_title("Distribution by Drone Type")
axes[1, 1].set_xlabel("Drone Type")
axes[1, 1].set_ylabel("Number of Samples")

# Adjust layout to avoid overlap of titles and labels
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
