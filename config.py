# config.py
import os

# Global configuration constants
DATA_DIR = os.path.join("input-dataset")
TEST_DATA_DIR = os.path.join("test-dataset")
OUTPUT_BASE_DIR = "."  # Base output directory; each year’s outputs go in a subfolder
YEARS = list(range(2007, 2016))
RANDOM_SEED = 42