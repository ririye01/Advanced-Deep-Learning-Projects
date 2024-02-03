import os
import sys

def read_file(file_path):
    # TODO: Write documentation (i.e. docstring)
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return [line.strip() for line in lines]
    

def create_directory(folder_path):
    # TODO: Write documentation (i.e. docstring)
    try:
        # Check if the directory already exists
        if not os.path.exists(folder_path):
            # Create the directory
            os.makedirs(folder_path)
            print(f"Directory created at: {folder_path}")
        else:
            print(f"Directory already exists at: {folder_path}")

        # Return the absolute path to the created directory
        return os.path.abspath(folder_path)
    except Exception as e:
        print(f"Error creating directory: {e}")
        return None