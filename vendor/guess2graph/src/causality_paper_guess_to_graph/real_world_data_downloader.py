"""
Real-World Data Downloader Module

Downloads real-world datasets from bnlearn library and saves them in the format
compatible with the experiment framework.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple
import bnlearn as bn


def download_dataset(dataset_name: str, output_dir: str = "real-world-data") -> None:
    """
    Download a single dataset and save in compatible format.
    
    Args:
        dataset_name: Name of the bnlearn dataset
        output_dir: Directory to save the data
    """
    print(f"Downloading {dataset_name}...")
    
    try:
        # Get data and adjacency matrix
        data_df = bn.import_example(dataset_name)
        model = bn.import_DAG(dataset_name)
        adj_matrix = model['adjmat'].values
        variable_names = list(data_df.columns)
        
        # Debug: Print original data types
        print(f"  Original data types: {dict(data_df.dtypes)}")
        print(f"  Sample values from first column: {data_df.iloc[:5, 0].tolist()}")
        
        # Convert categorical to numeric if needed
        for col in data_df.columns:
            if data_df[col].dtype == 'object':
                data_df[col] = pd.Categorical(data_df[col]).codes
                print(f"  Converted {col} from object to categorical codes")
        
        # Convert data to numpy array
        data = data_df.values
        print(f"  Data array dtype after conversion: {data.dtype}")
        print(f"  Kept discrete data as-is")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as tuple (adjacency_matrix, data, variable_names) in single .npy file
        output_path = os.path.join(output_dir, f"{dataset_name}-real-world-data.npy")
        np.save(output_path, (adj_matrix, data, variable_names))
        
        print(f"  Saved to {output_path}")
        print(f"  Shape: {data.shape}, Adjacency: {adj_matrix.shape}")
        print(f"  Variables: {variable_names}")
        
    except Exception as e:
        print(f"  Error downloading {dataset_name}: {str(e)}")


def download_datasets(dataset_names: List[str], output_dir: str = "real-world-data") -> None:
    """
    Download multiple datasets.
    
    Args:
        dataset_names: List of dataset names to download
        output_dir: Directory to save the data
    """
    print(f"Downloading {len(dataset_names)} datasets to {output_dir}/")
    print("-" * 60)
    
    for dataset_name in dataset_names:
        download_dataset(dataset_name, output_dir)
    
    print("-" * 60)
    print("Download complete!")


# Hardcoded list of datasets to download (modify as needed)
# DATASETS_TO_DOWNLOAD = [
#     'asia',
#     'sachs',
#     'alarm',
#     'child',
#     'insurance'
# ]
DATASETS_TO_DOWNLOAD = [
    'sachs'
]



if __name__ == "__main__":
    print(f"Datasets configured for download: {DATASETS_TO_DOWNLOAD}")
    print("To modify the list, edit the DATASETS_TO_DOWNLOAD variable in this script.")
    
    # Download the specified datasets
    download_datasets(DATASETS_TO_DOWNLOAD)