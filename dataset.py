import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# Load data
df_num = pd.read_csv("mnist_train.csv")
df_alph = pd.read_csv("A_Z Handwritten Data.csv")
df_alph["0"] += 10

# Set column names
pixel_array = ["Label"] + [f"pixel_{i}" for i in range(1, 785)]
df_num.columns = pixel_array
df_alph.columns = pixel_array

# Combine datasets
df = pd.concat([df_num, df_alph], axis=0)

# Prepare features and labels
X = df.drop(["Label"], axis=1).values.astype("float32").reshape(-1, 28, 28)
y = df["Label"].values.astype("int32")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Normalize data
X_train_mean = X_train.mean()
X_train_std = X_train.std()
X_test_mean = X_test.mean()
X_test_std = X_test.std()
X_train = (X_train - X_train_mean) / X_train_std
X_test = (X_test - X_test_mean) / X_test_std

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Save tensors
torch.save(X_train_tensor, 'X_train_tensor.pt')
torch.save(X_test_tensor, 'X_test_tensor.pt')
torch.save(y_train_tensor, 'y_train_tensor.pt')
torch.save(y_test_tensor, 'y_test_tensor.pt')

# Prepare DataLoader (using saved tensors)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)