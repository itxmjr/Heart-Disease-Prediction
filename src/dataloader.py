import os
import shutil
from pathlib import Path
import pandas as pd
import kagglehub
from dotenv import load_dotenv

load_dotenv()

def download_dataset(dataset_handle: str = "johnsmith88/heart-disease-dataset") -> Path:
    """
    Downloads the heart disease dataset using kagglehub and creates a symlink 
    in the project's data directory for easy access.
    """
    print(f"Downloading dataset: {dataset_handle}...")
    try:
        download_path = kagglehub.dataset_download(dataset_handle)
        print(f"Dataset downloaded to cache: {download_path}")
        
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data"
        
        data_dir.mkdir(parents=True, exist_ok=True)
        
        if data_dir.is_symlink() or data_dir.is_dir():
            print(f"Data already exists at {data_dir}. Removing old link/content...")
            if data_dir.is_symlink():
                data_dir.unlink()
            else:
                shutil.rmtree(data_dir)
        
        try:
            os.symlink(download_path, data_dir, target_is_directory=True)
            print(f"Created symlink: {data_dir} -> {download_path}")
        except OSError:
            print("Symlink failed, falling back to copying files...")
            shutil.copytree(download_path, data_dir)
            print(f"Copied files to: {data_dir}")
            
        return data_dir

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nMake sure you have Kaggle API credentials set up.")
        return None

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the heart disease dataset from CSV file.

    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully!")
        print(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return None
    
if __name__ == "__main__":
    path = download_dataset()
    if path:
        print(f"\nSuccess! Dataset available at: {path}")