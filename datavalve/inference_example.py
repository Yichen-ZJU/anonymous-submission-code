#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 DataValve  LoRA 

:
    python inference_example.py \
        --model-path /path/to/datavalve_ckpt/datavalve_xxx \
        --image-file /path/to/image.jpg \
        --prompt "Describe this image"
"""

import argparse
import torch
from PIL import Image
from transformers import AutoTokenizer
from peft import PeftModel

import sys
sys.path.append("./LLaVA")
from llava.model import LlavaLlamaForCausalLM
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

def load_model(model_path, base_model_path):
    """ LoRA """
    print(f" Base Model: {base_model_path}")
    model = LlavaLlamaForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print(f" LoRA: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    model = model.merge_and_unload()  #  LoRA 
    model.eval()

    print(f" Tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer

def inference(model, tokenizer, image_path, prompt):
    """"""
    image = Image.open(image_path).convert('RGB')
    prompt_with_image = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
    input_ids = tokenizer_image_token(
        prompt_with_image, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors='pt'
    ).unsqueeze(0).cuda()
    image_processor = model.get_vision_tower().image_processor
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(dtype=torch.bfloat16, device='cuda')
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=512,
            use_cache=True,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="")
    parser.add_argument("--base-model-path", type=str, 
                        default="./checkpoints/vicuna-7b-v1.5/",
                        help="Base model ")
    parser.add_argument("--image-file", type=str, required=True,
                        help="")
    parser.add_argument("--prompt", type=str, 
                        default="Describe this image in detail.",
                        help=" prompt")
    args = parser.parse_args()
    model, tokenizer = load_model(args.model_path, args.base_model_path)
    print(f"\n{'='*60}")
    print(f"Image: {args.image_file}")
    print(f"Prompt: {args.prompt}")
    print(f"{'='*60}\n")

    output = inference(model, tokenizer, args.image_file, args.prompt)

    print(f"Output:\n{output}")
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()

