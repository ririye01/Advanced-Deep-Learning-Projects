import sys
import os
import time
import argparse
import uuid
import torch
import json

from accelerate import PartialState
from diffusers import DiffusionPipeline

from utils import read_file, create_directory

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog='Stable-diffusion-pipeline.py',
        description='Generate images using stable diffusion'
    )

    parser.add_argument('model_id', help='The model id of a pretrained stable diffusion model hosted inside a model repo on huggingface.co.')
    parser.add_argument('prompt_file_path', help='The path to the files containing the prompts to use')
    parser.add_argument('-n', '--num_images', required=True, default=1, help='The number of images to generate per prompt. (Default: 1)')
    parser.add_argument('-o','--output_dir', required=True, help='Path to the folder where the results will be saved before being pushed to the Hugging face')
    parser.add_argument('negative_prompt_file_path', help='The path to the file containing the negative prompt to use when generating image')
    
    args = parser.parse_args()

    # Constants
    MODEL_ID            = args.model_id
    PROMPTS             = read_file(args.prompt_file_path)
    IMAGES_PER_PROMPT   = int(args.num_images)
    OUTPUT_DIR          = create_directory(args.output_dir)
    NEGATIVE_PROMPT     = read_file(args.negative_prompt_file_path)[0]
    NUM_GPUS            = torch.cuda.device_count()
    
    # Create the Stable diffusion pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, use_safetensors=True
    )

    # Distribute model across available GPUs
    distributed_state = PartialState()
    pipeline.to(distributed_state.device)

    # TODO: Log that generation started
    for prompt_idx, prompt in enumerate(PROMPTS):
        with distributed_state.split_between_processes([prompt] * NUM_GPUS) as prompts:
            
            pipeline_output = pipeline(prompt=prompts,
                                       negative_prompt=[NEGATIVE_PROMPT]*len(prompts),
                                       guidance_scale=9,
                                       num_images_per_prompt=IMAGES_PER_PROMPT // NUM_GPUS,
                                       return_dict=True)
            
            images = pipeline_output.images
            
            for image in images:
                # Generate a unique filename using uuid
                image_filename = f"{uuid.uuid4()}.png"
                
                # Save the image to output folder
                image.save(os.path.join(OUTPUT_DIR, image_filename))
                
                #  Add a new json, to "metadata.jsonl".
                metadata_entry = {
                    "file_name": image_filename,
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                }

                # Check if temp.jsonl file exists, append to it, or create a new one
                temp_file_path = os.path.join(OUTPUT_DIR, "temp.jsonl")
                mode = 'a' if os.path.exists(temp_file_path) else 'w'
                with open(temp_file_path, mode) as metadata_file:
                    metadata_file.write(json.dumps(metadata_entry) + '\n')

if __name__ == '__main__':
    main()