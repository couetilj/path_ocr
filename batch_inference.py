#!/usr/bin/env python
"""
For the future, if we need:

SmolVLM Batch Inference Script for OCR Training Dataset Generation

This script processes all images in a specified directory with SmolVLM-256M-Instruct
model and saves the extracted text to an annotations JSON file. It preserves any 
existing ground truth annotations in the file.

Usage:
    python batch_inference.py --image_dir images --output annotations.json

Requirements:
    - torch
    - transformers
    - PIL

Example:
    python batch_inference.py --image_dir images --output annotations.json
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
import time

# Default model to use
HUGGINGFACE_MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"

def load_annotations(file_path):
    """Load existing annotations from a JSON file"""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {file_path}. Creating a new annotations file.")
                return {}
    return {}

def save_annotations(annotations, output_file):
    """Save annotations to a JSON file"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=4)
    
    print(f"Annotations saved to {output_file}")

def extract_text_with_vlm(image_path, processor, model, device, prompt_text="Extract all visible text from this image."):
    """Process an image with SmolVLM to extract text"""
    try:
        # Load image
        image = Image.open(image_path)
        
        # Create input messages for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]
            },
        ]
        
        # Prepare inputs
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = inputs.to(device)
        
        # Generate output
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=500)
        
        # Decode the output
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        
        # Extract only the assistant's response
        if "ASSISTANT:" in generated_text:
            generated_text = generated_text.split("ASSISTANT:", 1)[1].strip()
        
        # Add metadata with model name, timestamp and prompt used
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata = f"<model>{HUGGINGFACE_MODEL_NAME}</model><time>{timestamp}</time><prompt>{prompt_text}</prompt>\n\n"
        
        # Return the text with metadata
        text_to_return = metadata + generated_text
        
        # Clean up any lingering template text
        if "Assistant:" in text_to_return:
            text_to_return = text_to_return.split("Assistant:", 1)[1].strip()
        
        return text_to_return
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return f"Error: {str(e)}"

def load_vlm_model(model_name):
    """Load the VLM model and processor"""
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        
        print(f"Loading processor from {model_name}...")
        processor = AutoProcessor.from_pretrained(model_name)
        
        # Device selection - CUDA if available, otherwise CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        print(f"Loading model from {model_name}...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            _attn_implementation="flash_attention_2" if device == "cuda" else "eager",
        ).to(device)
        
        return processor, model, device
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None, None

def get_image_files(directory_path):
    """Get all image files from a directory"""
    image_files = []
    for file in os.listdir(directory_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_files.append(os.path.join(directory_path, file))
    return image_files

def process_images(image_dir, output_file, model_name=HUGGINGFACE_MODEL_NAME, batch_size=1):
    """Process all images in a directory and save annotations"""
    # Load existing annotations
    annotations = load_annotations(output_file)
    
    # Load model
    processor, model, device = load_vlm_model(model_name)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Get all image files
    image_files = get_image_files(image_dir)
    print(f"Found {len(image_files)} images in {image_dir}")
    
    # Count images that need processing (don't have model_extracted_text)
    to_process = [img for img in image_files 
                  if img not in annotations or 
                  not annotations[img].get("model_extracted_text")]
    
    print(f"{len(to_process)} images need text extraction")
    
    if not to_process:
        print("No images need processing. Exiting.")
        return
    
    # Process images
    with tqdm(total=len(to_process), desc="Processing images") as pbar:
        for img_path in to_process:
            # Check if we already have ground truth for this image
            existing_entry = annotations.get(img_path, {})
            
            # Only extract text if we don't have it already
            extracted_text = extract_text_with_vlm(img_path, processor, model, device)
            
            # Update the annotation entry
            annotations[img_path] = {
                "model_extracted_text": extracted_text,
                # Preserve existing ground truth if present
                "corrected_text": existing_entry.get("corrected_text", "")
            }
            
            # Save after each image to ensure we don't lose progress if something fails
            save_annotations(annotations, output_file)
            
            # Update progress bar
            pbar.update(1)
            
            # Small delay to prevent overloading the GPU
            time.sleep(0.1)
    
    print(f"Completed processing {len(to_process)} images.")
    print(f"Final annotations saved to {output_file}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process images with SmolVLM for OCR training dataset generation")
    parser.add_argument("--image_dir", type=str, default="images", help="Directory containing images to process")
    parser.add_argument("--output", type=str, default="annotations.json", help="Output JSON file for annotations")
    parser.add_argument("--model", type=str, default=HUGGINGFACE_MODEL_NAME, help="HuggingFace model name to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing (usually 1 for memory constraints)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process images
    process_images(args.image_dir, args.output, args.model, args.batch_size)

if __name__ == "__main__":
    main()