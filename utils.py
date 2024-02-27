from pytorch_lightning import Callback
from torch.utils.data import Subset
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
import pandas as pd


class PrintMetricsCallback(Callback):
    def __init__(self, print_epoch=1):
        """
        Args:
            print_epoch (int): Frequency of epochs to print metrics. Default is 1, meaning print every epoch.
        """
        super().__init__()
        self.print_epoch = print_epoch

    # seems like the print should be at the end of the validation epoch
    # but the validation logs are getting flushed before the training logs
    # this might vary depending on the model or ???

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch + 1  # Adjust for human-readable epoch numbering
        if current_epoch % self.print_epoch == 0:
            logs = trainer.logged_metrics
            metrics_string = ", ".join([f"{key}: {value:0.4f}" for key, value in logs.items()])
            print(f"Epoch {current_epoch} Metrics: {metrics_string}")

    def on_fit_start(self, trainer, pl_module):
        print(f'Beginning training for at most {trainer.max_epochs} epochs')

    def on_fit_end(self, trainer, pl_module):
        print(f'End.  Trained for {trainer.current_epoch} epochs.')

def denormalize(tensor, mean, std):
    """
    Denormalizes a tensor given the mean and std used for normalization.
    This function supports tensors for single-channel and multi-channel images.
    
    - tensor: Input tensor with shape (C, H, W) or (N, C, H, W).
    - mean: The mean used for normalization (per channel).
    - std: The standard deviation used for normalization (per channel).
    
    Both mean and std arguments should be sequences (e.g., lists or tuples) or scalars,
    with the length equal to the number of channels in the tensor.
    """
    if not isinstance(mean, (list, tuple)):
        mean = [mean]
    if not isinstance(std, (list, tuple)):
        std = [std]
    
    if len(tensor.shape) == 3:  # Single image (C, H, W)
        mean = torch.tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
        std = torch.tensor(std, dtype=tensor.dtype).view(-1, 1, 1)
    elif len(tensor.shape) == 4:  # Batch of images (N, C, H, W)
        mean = torch.tensor(mean, dtype=tensor.dtype).view(1, -1, 1, 1)
        std = torch.tensor(std, dtype=tensor.dtype).view(1, -1, 1, 1)
    
    return tensor * std + mean

def sample_dataset(dataset, num_samples=1000, random_state=42):
    # for reproducibility
    np.random.seed(random_state)

    # Determine the number of classes
    num_classes = len(dataset.classes)

    # Initialize a list to store indices for each class
    indices_per_class = {class_idx: [] for class_idx in range(num_classes)}

    # Go through the dataset and store indices of each class
    for idx, (_, class_idx) in enumerate(dataset):
        indices_per_class[class_idx].append(idx)

    # Sample indices for each class
    sampled_indices = []
    for class_idx, indices in indices_per_class.items():
        if len(indices) > num_samples:
            sampled_indices.extend(np.random.choice(indices, num_samples, replace=False))
        else:
            sampled_indices.extend(indices)

    sampled_dataset = Subset(dataset, sampled_indices)

    return sampled_dataset

def center_crop_and_resize(image_path, output_size):
    """
    Crop an image to the largest possible center square and then resize it to a specified square size.
    
    Parameters:
    - image_path (str): The file path of the image to be processed.
    - output_size (int): The width and height of the output image in pixels. The output image will be a square,
      so only one dimension is needed.
    
    Returns:
    - Image: A PIL Image object of the cropped and resized image.
    
    This function first calculates the largest square that can be cropped from the center of the original image.
    It then crops the image to this square and resizes the cropped image to the specified dimensions.
    """
    
    # Open the image
    image = Image.open(image_path)
    
    # Calculate the dimensions for a center square crop
    width, height = image.size
    crop_size = min(width, height)
    
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = (width + crop_size) / 2
    bottom = (height + crop_size) / 2
    
    # Crop the center of the image
    image_cropped = image.crop((left, top, right, bottom))
    
    # Resize the cropped image
    image_resized = image_cropped.resize((output_size, output_size), Image.Resampling.LANCZOS)
    
    return image_resized



class Visualizer():
    def __init__(self, dataset, image_extractor=None, label_extractor=None):
        self.dataset = list(dataset)
        self.image_extractor = image_extractor
        self.label_extractor = label_extractor
        
    def visualize(self, rows=4, cols=4, figsize=(12, 12), sample=False):
        # Visualizer
        fig=plt.figure(figsize=figsize)
        for i in range(1, rows * cols + 1):
            if sample:
                data = random.choice(self.dataset)
            else:
                data = self.dataset[i-1]
            if self.image_extractor:
                img = self.image_extractor(data)
            else:
                img = data
                
            # An image can be a string (filepath), numpy array, PIL Image
            ax = fig.add_subplot(rows, cols, i)
            #ax.set_yticklabels([])
            #ax.set_xticklabels([])
            ax.set_axis_off()
            
            if self.label_extractor:
                label = self.label_extractor(data)
                ax.set_title(label)
                
            if type(img) == Image:
                img = np.array(img)
            if type(img) == torch.Tensor:
                img = img.numpy()
            plt.imshow(img)
        plt.show()

def process_experiment_logs(logs_path, exper_name):
    """
    Processes the latest version of the experiment logs, groups metrics by epoch,
    and adds an experiment name column.  
    
    Parameters:
    - logs_path: Path to the directory containing CSV logs.  
    - legend_name: Name of the experiment for plotting legend.

    Returns:
    - A pandas DataFrame with metrics grouped by epoch and an added 'exp_name' column.
    """
    log_dir = os.path.join(logs_path, "model")
    versions = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    latest_version = sorted(versions, key=lambda x: int(x.split('_')[-1]))[-1]
    latest_log_path = os.path.join(log_dir, latest_version, "metrics.csv")

    # Read the CSV into a DataFrame
    df = pd.read_csv(latest_log_path)

    # Drop the 'step' column if it exists
    if 'step' in df.columns:
        df = df.drop(columns=['step'])

    # Group by 'epoch' and aggregate the metrics
    df_grouped = df.groupby('epoch', as_index=False).mean()

    # Add the 'exp_name' column
    df_grouped['exp_name'] = exper_name

    return df_grouped