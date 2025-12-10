from huggingface_hub import snapshot_download
import os

print("Downloading OpenWebText corpus from Hugging Face...")
print("This may take a while (corpus is ~38GB of text data)...")

# Create data directory
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "openwebtext")
os.makedirs(data_dir, exist_ok=True)

# Download the dataset files directly using huggingface_hub
# This downloads all the parquet/arrow files
local_path = snapshot_download(
    repo_id="Skylion007/openwebtext",
    repo_type="dataset",
    local_dir=data_dir,
    resume_download=True
)

print(f"\nDownload complete!")
print(f"Dataset downloaded to: {local_path}")
print("\nFiles downloaded:")
for root, dirs, files in os.walk(local_path):
    for file in files:
        filepath = os.path.join(root, file)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  {file}: {size_mb:.2f} MB")
