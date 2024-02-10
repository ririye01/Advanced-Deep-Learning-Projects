from datasets import load_dataset

import os


def download_huggingface_data(
    dataset_path: str,
    output_folder: str,
) -> None:
    """
    Download all of the data for images created by stable diffusion.

    Paramethers
        dataset_path (str): HuggingFace data path, typically in format ("<author or org name>/<dataset card id>")
        output_folder (str): Exact directory where images will be saved to.
    """
    dataset = load_dataset(dataset_path, split="train")
    os.makedirs(output_folder, exist_ok=True)
    dataset.to_parquet(f"{output_folder}/reference_diffusion_dataset.parquet")


if __name__ == "__main__":
    for stable_diffusion_model in ["sd1.5", "sdxl"]:
        download_huggingface_data(
            dataset_path=f"CS-8321/{stable_diffusion_model}-images",
            output_folder=f"../data/{stable_diffusion_model}-images",
        )
