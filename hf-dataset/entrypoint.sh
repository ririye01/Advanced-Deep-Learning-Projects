#!/bin/bash

apt-get -qq update && apt-get -qq install --no-install-recommends -y libgl1-mesa-glx libglib2.0-0

pip install -r ./requirements.txt

accelerate launch --num_processes=4 stable-diffusion-pipeline.py "runwayml/stable-diffusion-v1-5" prompts.txt -n 256 -o /output/ negative_prompt.txt

# python update_metadata_using_clip.py /output/temp.jsonl /output/

# python to_hf.py