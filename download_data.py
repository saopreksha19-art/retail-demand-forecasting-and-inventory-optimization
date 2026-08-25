import kagglehub
import shutil
import os

DATA_DIR = "data/raw"

files = [
    "sales_train_validation.csv",
    "calendar.csv",
    "sell_prices.csv"
]

# Download the dataset
dataset_path = kagglehub.dataset_download(
    "aryayadav0513/m5-forecasting-accuracy"
)

print("Dataset downloaded to:")
print(dataset_path)

# Create our project data directory
os.makedirs(DATA_DIR, exist_ok=True)

# Copy required files into our project
for file in files:
    source = os.path.join(dataset_path, file)
    destination = os.path.join(DATA_DIR, file)

    if os.path.exists(source):
        shutil.copy2(source, destination)
        print(f"Copied: {file}")
    else:
        print(f"WARNING: {file} not found")

print("\nFiles in data/raw:")
for file in os.listdir(DATA_DIR):
    print(file)
