import os
import shutil

def check_num_files(directory, expected_num_files):
    actual_num_files = sum(len(files) for _, _, files in os.walk(directory))
    if actual_num_files != expected_num_files:
        print(f"Warning: Expected {expected_num_files} files, but found {actual_num_files}.")
    else:
        print("File count is as expected.")
    return actual_num_files == expected_num_files

def dirreworker(directory):
    """
    Reorganizes the directory structure to ensure it has a consistent format.
    This function is a placeholder and should be implemented based on specific requirements.
    """
    # Implement the reorganization logic here
    print(f"Reorganizing directory: {directory}")
    # Example: Move files, create subdirectories, etc.

    for class_folder in os.listdir(directory):
        class_path = os.path.join(directory, class_folder)
        if not os.path.isdir(class_path):
            continue
        
        for file in os.listdir(class_path):
            shutil.move(
                os.path.join(class_path, file),
                os.path.join(directory, file)
            )
        os.rmdir(class_path)  # Remove empty class directories after moving files
    print("Directory reorganization complete.")

if __name__ == "__main__":
    directory = "./TFG_Dataset/train/images/"
    expected_num_files = 6988  # Set your expected number of files
    #check_num_files(directory, expected_num_files)
    #dirreworker("./TFG_Dataset/train/images/")
    #dirreworker("./TFG_Dataset/train/labels/")
    #dirreworker("./TFG_Dataset/val/images/")
    dirreworker("./TFG_Dataset/val/labels/")
    dirreworker("./TFG_Dataset/test/images/")
    dirreworker("./TFG_Dataset/test/labels/")