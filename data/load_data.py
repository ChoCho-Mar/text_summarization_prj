from datasets import load_dataset
import pandas as pd
import os

# Using logging instead of print for better practice in larger applications
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the dataset
logger.info("Loading dataset...")

try:
    dataset = load_dataset('samsum')
    logger.info("Dataset loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    raise

# Datasets to process
datasets = ['train', 'test', 'validation']

# Process each dataset
for ds in datasets:
    logger.info(f"Processing {ds} dataset...")
    try:
        data = dataset[ds]
        df = pd.DataFrame(data)
        
        # Save to CSV
        file_name = f'{ds}_dataset.csv'
        
        if os.path.exists(file_name):
            logger.warning(f"{file_name} already exists, overwriting.")
        
        df.to_csv(file_name, index=False)
        logger.info(f"{file_name} saved successfully!")
    except Exception as e:
        logger.error(f"Error processing {ds} dataset: {e}")

logger.info("Dataset processing complete.")