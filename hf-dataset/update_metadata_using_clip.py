import json
import argparse
import os
from PIL import Image
from clip_interrogator import Config, Interrogator

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

# Input and output file paths
temp_jsonl_path = "temp.jsonl"
output_jsonl_path = "metadata.jsonl"

# Importing CLIP interrogation functions
caption_model_name = 'blip-large'
clip_model_name = 'ViT-L-14/openai'
config = Config()
config.clip_model_name = clip_model_name
config.caption_model_name = caption_model_name
ci = Interrogator(config)

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
    with open(TEMP_JSONL_PATH, 'r') as temp_file, open(os.path.join(OUTPUT_DIR, "metadata.jsonl"), 'w') as output_file:
        for line in temp_file:
            # Process each line and write updated metadata to metadata.jsonl
            updated_metadata_entry = interrogate_and_update_metadata(line, OUTPUT_DIR)
            output_file.write(json.dumps(updated_metadata_entry) + '\n')

    # Remove temp.jsonl file
    # os.remove(TEMP_JSONL_PATH)

if __name__ == "__main__":
    main()
