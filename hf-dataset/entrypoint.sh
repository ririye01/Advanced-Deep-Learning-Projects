#!/bin/bash

apt-get -qq update && apt-get -qq install --no-install-recommends -y libgl1-mesa-glx libglib2.0-0

pip install -r ./requirements.txt

# TODO: Launch the distributed pipeline using accelerate
accelerate launch --num_processes=2 stable-diffusion-pipeline.py "runwayml/stable-diffusion-v1-5" prompts.txt -n 4 -o /output/ negative_prompt.txt