import os
import tarfile
import lzma
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "openwebtext")
SUBSETS_DIR = os.path.join(DATA_DIR, "subsets")
EXTRACTED_DIR = os.path.join(DATA_DIR, "extracted")

def extract_xz_file(xz_path, output_dir):
    """Extract a single .xz file to text"""
    try:
        # Create output filename (remove .xz extension)
        basename = os.path.basename(xz_path).replace('.xz', '.txt')
        output_path = os.path.join(output_dir, basename)
        
        # Skip if already extracted
        if os.path.exists(output_path):
            return True
        
        # Decompress .xz to text
        with lzma.open(xz_path, 'rt', encoding='utf-8') as f_in:
            content = f_in.read()
        
        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.write(content)
        
        # Remove .xz file after extraction to save space
        os.remove(xz_path)
        return True
    except Exception as e:
        print(f"Error extracting {xz_path}: {e}")
        return False

def process_tar_file(tar_path):
    """Extract a single tar file and decompress its contents"""
    tar_name = os.path.basename(tar_path).replace('.tar', '')
    output_dir = os.path.join(EXTRACTED_DIR, tar_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nExtracting: {tar_name}")
    
    # Extract tar file
    temp_dir = os.path.join(DATA_DIR, "temp_extract")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(temp_dir)
        
        # Find all .xz files
        xz_files = glob.glob(os.path.join(temp_dir, "**", "*.xz"), recursive=True)
        
        # Decompress each .xz file
        for xz_path in tqdm(xz_files, desc=f"Decompressing {tar_name}", leave=False):
            extract_xz_file(xz_path, output_dir)
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True
    except Exception as e:
        print(f"Error processing {tar_path}: {e}")
        return False

def main():
    print("=" * 60)
    print("OpenWebText Corpus Extraction")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(EXTRACTED_DIR, exist_ok=True)
    
    # Find all tar files
    tar_files = sorted(glob.glob(os.path.join(SUBSETS_DIR, "*.tar")))
    print(f"\nFound {len(tar_files)} tar files to extract")
    
    if not tar_files:
        print("No tar files found! Make sure the corpus is downloaded.")
        return
    
    # Process each tar file sequentially (to manage disk space)
    for i, tar_path in enumerate(tar_files, 1):
        print(f"\n[{i}/{len(tar_files)}] Processing {os.path.basename(tar_path)}")
        process_tar_file(tar_path)
    
    # Count total extracted files
    total_files = 0
    for root, dirs, files in os.walk(EXTRACTED_DIR):
        total_files += len([f for f in files if f.endswith('.txt')])
    
    print("\n" + "=" * 60)
    print("Extraction Complete!")
    print(f"Total text files extracted: {total_files}")
    print(f"Location: {EXTRACTED_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
