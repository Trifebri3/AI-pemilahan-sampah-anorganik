import os
import shutil
import random

# Set random seed for reproducibility
random.seed(42)

def split_dataset():
    src_dir = "dataset"
    dest_dir = "dataset_split"
    
    # Clear legacy split directory if it exists
    if os.path.exists(dest_dir):
        print(f"Cleaning existing split directory: {dest_dir}")
        shutil.rmtree(dest_dir)
        
    classes = ["Kaca", "Kertas", "Logam", "Plastik", "Residu"]
    splits = ["train", "val", "test"]
    
    # Ratios
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15
    
    print("=== Dataset Splitting Process ===")
    
    # Create target directories
    for split in splits:
        for cls in classes:
            path = os.path.join(dest_dir, split, cls)
            os.makedirs(path, exist_ok=True)
            
    total_copied = 0
    
    for cls in classes:
        cls_src_path = os.path.join(src_dir, cls)
        if not os.path.exists(cls_src_path):
            print(f"Warning: Directory {cls_src_path} does not exist!")
            continue
            
        images = [f for f in os.listdir(cls_src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
        random.shuffle(images)
        
        num_images = len(images)
        num_train = int(num_images * train_ratio)
        num_val = int(num_images * val_ratio)
        num_test = num_images - num_train - num_val  # ensure they sum to total
        
        train_imgs = images[:num_train]
        val_imgs = images[num_train:num_train + num_val]
        test_imgs = images[num_train + num_val:]
        
        print(f"Class: {cls} | Total: {num_images} -> Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")
        
        # Copy helper
        def copy_files(file_list, split_name):
            copied = 0
            for img in file_list:
                src_file = os.path.join(cls_src_path, img)
                dest_file = os.path.join(dest_dir, split_name, cls, img)
                shutil.copy2(src_file, dest_file)
                copied += 1
            return copied

        copied_train = copy_files(train_imgs, "train")
        copied_val = copy_files(val_imgs, "val")
        copied_test = copy_files(test_imgs, "test")
        
        total_copied += (copied_train + copied_val + copied_test)
        
    print(f"\nDone! Total images copied and structured: {total_copied}")
    print("Structure saved in 'dataset_split/' folder.")

if __name__ == "__main__":
    split_dataset()
