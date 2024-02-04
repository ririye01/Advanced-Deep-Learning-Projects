import json
import logging
import argparse
import os
from PIL import Image
from clip_interrogator import Config, Interrogator


# ----------------------------LOGGING SETUP------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

main_logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")

main_logger = logging.getLogger(__name__)
main_logger.setLevel(logging.INFO)
main_file_handler = logging.FileHandler("clip-interrogation.log")
main_file_handler.setFormatter(formatter)
main_logger.addHandler(main_file_handler)

# -----------------------------------------------------------------------------

# ----------------------CLIP INTERROGATION SETUP-------------------------------
caption_model_name = 'blip-large'
clip_model_name = 'ViT-L-14/openai'
config = Config()
config.clip_model_name = clip_model_name
config.caption_model_name = caption_model_name
ci = Interrogator(config)
# -----------------------------------------------------------------------------

# Function to interrogate CLIP and update metadata
def interrogate_and_update_metadata(line, output_dir):
    # Decode JSON
    metadata_entry = json.loads(line)

    # Get the file path associated with the file_name key
    file_path = os.path.join(output_dir, metadata_entry["file_name"])

    # Create a PIL image
    image = Image.open(file_path)

    # Interrogate CLIP
    clip_interrogation = image_to_prompt(image, 'best')

    # Add a new field called "interrogation" to the json
    metadata_entry["interrogation"] = clip_interrogation

    return metadata_entry

def image_to_prompt(image, mode):
    ci.config.chunk_size = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    ci.config.flavor_intermediate_count = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    image = image.convert('RGB')
    if mode == 'best':
        return ci.interrogate(image)
    elif mode == 'classic':
        return ci.interrogate_classic(image)
    elif mode == 'fast':
        return ci.interrogate_fast(image)
    elif mode == 'negative':
        return ci.interrogate_negative(image)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog='update_metadata_using_clip.py',
        description='Generate clip interrogations for images'
    )

    parser.add_argument('temp_jsonl_path', help='The model id of a pretrained stable diffusion model hosted inside a model repo on huggingface.co.')
    parser.add_argument('output_dir', help='The path to the files containing the prompts to use')
    
    args = parser.parse_args()

    TEMP_JSONL_PATH = args.temp_jsonl_path
    OUTPUT_DIR = args.output_dir

    # Open temp.jsonl for reading and metadata.jsonl for writing
    metadata_file_path = os.path.join(OUTPUT_DIR, "metadata.jsonl")
    mode = 'a' if os.path.exists(metadata_file_path) else 'w'
    
    # Load the processed line number from a file, or start from the beginning
    last_processed_line_file = os.path.join(OUTPUT_DIR, "last_processed_line.txt")
    start_line = 1

    if os.path.exists(last_processed_line_file):
        with open(last_processed_line_file, 'r') as last_processed_file:
            content = last_processed_file.read().strip()
            if content:
                try:
                    start_line = int(content) + 1
                except ValueError:
                    main_logger.warning(f"Error: Invalid content in last_processed_line.txt. Starting from the beginning.")
                    start_line = 1

    with open(TEMP_JSONL_PATH, 'r') as temp_file, open(metadata_file_path, mode) as output_file:     

        for i, line in enumerate(temp_file, start=1):
            if i < start_line:
                main_logger.info(f"Skipping line {i} in temp.jsonl.")
                continue

            # Process each line and write updated metadata to metadata.jsonl
            updated_metadata_entry = interrogate_and_update_metadata(line, OUTPUT_DIR)
            output_file.write(json.dumps(updated_metadata_entry) + '\n')
            main_logger.info(f"Successfully generated CLIP interrogation for image {i}")

            # Update the last processed line number after writing each line
            with open(last_processed_line_file, 'w') as last_processed_file:
                last_processed_file.write(str(i))
            
            main_logger.info(f"'last_processed_line.txt' was updated. The last processed line is now {i}.")

    # Remove temp.jsonl file
    # os.remove(TEMP_JSONL_PATH)

if __name__ == "__main__":
    main()
