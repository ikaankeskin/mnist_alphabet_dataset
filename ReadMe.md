# Instructions for Using the Dataset

This guide will assist you in extracting dataset contents, running the `dataset.py` script, and loading the saved `.pt` files for training in a PyTorch environment.

## Steps:

1. **Extracting .zip Files:**
   - Ensure that the .zip files are located in the same directory as `dataset.py`.
   - Use your preferred method (such as a file extractor or a command line tool) to extract the contents of the .zip files.

2. **Running `dataset.py`:**
   - Navigate to the directory containing `dataset.py`.
   - Run the script by executing `python dataset.py` in your command line or terminal. This script will process the data and save tensors in `.pt` format.

3. **Loading Data for Training:**
   - In your training script, import the necessary PyTorch modules:
     ```python
     import torch
     from torch.utils.data import TensorDataset, DataLoader
     ```
   - Load the saved `.pt` files as follows:
     ```python
     X_train_tensor = torch.load('X_train_tensor.pt')
     X_test_tensor = torch.load('X_test_tensor.pt')
     y_train_tensor = torch.load('y_train_tensor.pt')
     y_test_tensor = torch.load('y_test_tensor.pt')
     ```
   - Create a `TensorDataset` and a `DataLoader` for the training data:
     ```python
     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
     dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
     ```

## Additional Notes:

- Ensure `batch_size` is set to a suitable value according to your system's capabilities and the size of your dataset.
- The number of workers (`num_workers`) in the `DataLoader` can be adjusted based on your system's CPU capabilities. More workers can speed up data loading but require more CPU resources.
- It's important to verify that the paths to the `.pt` files are correct and accessible from your training script.

For any further assistance or queries, feel free to reach out or consult the PyTorch documentation for detailed guidelines on using `DataLoader` and `TensorDataset`.
