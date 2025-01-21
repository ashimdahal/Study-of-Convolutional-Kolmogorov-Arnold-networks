import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data dictionary with all models and their metrics
data = {
    'AlexNet KAN': {
        'top5_acc': 67.72,
        'top1_acc': 42.79,
        'loss': 2.62,
        'inference_time': 0.0074,
        'flops': 1611568352,
        'training_time': 48*24*3600,  # converting days to seconds
        'parameters': 39756776
    },
    'AlexNet': {
        'top5_acc': 79.07,
        'top1_acc': 56.62,
        'loss': 2.31,
        'inference_time': 0.0018,
        'flops': 714197696,
        'training_time': 3*24*3600,  # converting days to seconds
        'parameters': 61100840
    },
    'LeNet KAN': {
        'top5_acc': np.nan,
        'top1_acc': 98.81,
        'loss': 0.036,
        'inference_time': 0.003,
        'flops': 3298728,
        'training_time': 981.45,
        'parameters': 82128
    },
    'LeNet': {
        'top5_acc': np.nan,
        'top1_acc': 98.89,
        'loss': 0.031,
        'inference_time': 0.0007,
        'flops': 429128,
        'training_time': 888.77,
        'parameters': 61750
    },
    'Tabular CNN': {
        'top5_acc': np.nan,
        'top1_acc': 47.61,
        'loss': 0.0167,
        'inference_time': 0.00004,
        'flops': 37853010,
        'training_time': 6450.5,
        'parameters': 7818482
    },
    'Tabular CKAN': {
        'top5_acc': np.nan,
        'top1_acc': 45.09,
        'loss': 0.0172,
        'inference_time': 0.0001,
        'flops': 79861586,
        'training_time': 10646.07,
        'parameters': 6265998
    }
}

# Convert to DataFrame
df = pd.DataFrame(data).T

# Function to normalize values between 0 and 1
def normalize_column(col, inverse=False):
    min_val = col.min()
    max_val = col.max()
    if inverse:
        return 1 - ((col - min_val) / (max_val - min_val))
    return (col - min_val) / (max_val - min_val)

# Select metrics for visualization
metrics = ['top1_acc', 'inference_time', 'flops', 'parameters', 'loss']
normalized_df = pd.DataFrame()

# Normalize each metric (inverse for metrics where lower is better)
normalized_df['Accuracy'] = normalize_column(df['top1_acc'])
normalized_df['Speed'] = normalize_column(df['inference_time'], inverse=True)
normalized_df['Efficiency'] = normalize_column(df['flops'], inverse=True)
normalized_df['Compactness'] = normalize_column(df['parameters'], inverse=True)
normalized_df['Performance'] = normalize_column(df['loss'], inverse=True)

# Number of variables
categories = list(normalized_df.columns)
N = len(categories)

# Create the angle values for the radar chart
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Create the plot
plt.figure(figsize=(12, 8))
ax = plt.subplot(111, polar=True)

# Plot data
for idx, model in enumerate(normalized_df.index):
    values = normalized_df.loc[model].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
    ax.fill(angles, values, alpha=0.1)

# Fix axis to go in the right order and start at 12 o'clock
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Draw axis lines for each angle and label
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.title("Model Comparison Radar Plot\nHigher values indicate better performance", pad=20)

# Add gridlines
ax.grid(True)

# Ensure the plot is shown as a circle by setting aspect ratio to 'equal'
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig("graphs/radar_plot.png")
plt.show()
