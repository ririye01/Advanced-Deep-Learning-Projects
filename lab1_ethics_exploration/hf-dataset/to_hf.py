"""Push dataset to Hugging face"""
from datasets import load_dataset

if __name__ == '__main__':

    # Load the dataset
    dataset = load_dataset("imagefolder", data_dir="/output")

    # Push it to the hub
    dataset.push_to_hub("CS-8321/demo_dataset")