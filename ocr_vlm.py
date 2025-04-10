import streamlit as st
import os
import json
from PIL import Image
import pandas as pd
from pathlib import Path
import time

# Which VLM model to use?
HUGGINGFACE_MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"

# Set page configuration
st.set_page_config(
    page_title="OCR Training Dataset Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title
st.title("OCR Training Dataset Generator")

# Flag to track if VLM dependencies are available
vlm_available = False

# Try to import the necessary packages for the VLM model
try:
    import torch
    # To suppress warnings from pytorch
    torch.classes.__path__ = []
    from transformers import AutoProcessor, AutoModelForVision2Seq
    vlm_available = True
except ImportError:
    vlm_available = False
    
# VLM Model loading function
@st.cache_resource
def load_vlm_model():
    """Load the SmolVLM model and processor (cached to avoid reloading)"""
    try:
        # Device selection - CUDA if available, otherwise CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        with st.status("Loading SmolVLM model..."):
            # Load processor and model
            processor = AutoProcessor.from_pretrained(HUGGINGFACE_MODEL_NAME)
            
            # Load with appropriate settings for device
            model = AutoModelForVision2Seq.from_pretrained(
                HUGGINGFACE_MODEL_NAME,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                _attn_implementation="flash_attention_2" if device == "cuda" else "eager",
            ).to(device)
            
            return processor, model, device
    except Exception as e:
        st.error(f"Error loading SmolVLM model: {str(e)}")
        return None, None, None

# Function to extract text from image using SmolVLM
def extract_text_with_vlm(image_path, processor, model, device):
    """Process an image with SmolVLM to extract text"""
    try:
        # Load image
        image = Image.open(image_path)
        
        # Define the prompt
        prompt_text = "Extract all visible text from this image."
        
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
        
        # Extract only the actual response content by removing template elements
        if "ASSISTANT:" in generated_text:
            generated_text = generated_text.split("ASSISTANT:", 1)[1].strip()
        
        # Add metadata with model name, timestamp and prompt used
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata = f"<model>{HUGGINGFACE_MODEL_NAME}</model><time>{timestamp}</time><prompt>{prompt_text}</prompt>\n\n"
        
        # Return only the actual text content with metadata
        text_to_return = metadata + generated_text

        # Right now the response looks like this:
        # <model>HuggingFaceTB/SmolVLM-256M-Instruct</model><time>2025-03-22 10:35:00</time>
        # User:
        # Extract all visible text from this image. Assistant: A label with the letters SNXX-XXXX on it.
        # I only want to retain the tag and everything after "Assistant:"
        
        # So, if "Assistant:" is in the response, split the response at "Assistant:" and return everything after it
        if "Assistant:" in text_to_return:
            text_to_return = text_to_return.split("Assistant:", 1)[1].strip()
        
        return text_to_return
    except Exception as e:
        st.error(f"Error extracting text with VLM: {str(e)}")
        return f"Error: {str(e)}"

# Function to load images from a directory
def load_images_from_directory(directory_path):
    image_files = []
    for file in os.listdir(directory_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_files.append(os.path.join(directory_path, file))
    return image_files

# Function to save annotations
def save_annotations(annotations, output_file):
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=4)
    
    # (I used to have this code to export as CSV, but it's not needed for now)
    # # Also export as CSV for easy viewing
    # df = pd.DataFrame(annotations).T
    # df.reset_index(inplace=True)
    # df.rename(columns={'index': 'image_path'}, inplace=True)
    # csv_path = output_file.replace('.json', '.csv')
    # df.to_csv(csv_path, index=False)
    # return csv_path

# Function to load existing annotations
def load_annotations(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}

# Initialize session state variables
if 'annotations' not in st.session_state:
    st.session_state.annotations = {}

if 'vlm_model_loaded' not in st.session_state:
    st.session_state.vlm_model_loaded = False

# Directory selection
with st.sidebar:
    st.header("Settings")
    directory_path = st.text_input("Enter images directory path", "images")
    
    # Add VLM Settings section
    st.header("VLM Settings")
    
    # Show VLM status
    if vlm_available:
        st.success("VLM dependencies available")
        
        # Add button to load model
        if not st.session_state.vlm_model_loaded:
            if st.button("Load SmolVLM Model"):
                # Load model
                processor, model, device = load_vlm_model()
                if model is not None:
                    st.session_state.processor = processor
                    st.session_state.model = model
                    st.session_state.device = device
                    st.session_state.vlm_model_loaded = True
                    st.success("SmolVLM model loaded successfully")
                else:
                    st.error("Failed to load SmolVLM model")
        else:
            st.success("SmolVLM model loaded and ready")
            if st.button("Unload SmolVLM Model"):
                # Clean up model to free memory
                if hasattr(st.session_state, 'model'):
                    del st.session_state.model
                if hasattr(st.session_state, 'processor'):
                    del st.session_state.processor
                if hasattr(st.session_state, 'device'):
                    del st.session_state.device
                st.session_state.vlm_model_loaded = False
                # Force garbage collection
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                st.rerun()
    else:
        st.warning("VLM dependencies not available. Install required packages to use SmolVLM.")
        st.code("pip install torch transformers", language="bash")
    
    # Continue with directory loading
    if os.path.exists(directory_path):
        image_files = load_images_from_directory(directory_path)
        
        # Check if there are images in the directory
        if image_files:
            st.success(f"Found {len(image_files)} images in the directory.")
            
            # Output file selection
            output_file = st.text_input("Output JSON file", "annotations.json")
            
            # Load existing annotations if file exists
            if os.path.exists(output_file):
                st.session_state.annotations = load_annotations(output_file)
                st.info(f"Loaded {len(st.session_state.annotations)} existing annotations.")
            
            # Create thumbnails for navigation
            st.header("Image Navigation")
            
            # Display thumbnails in a grid
            num_cols = 3
            thumbnail_size = (100, 100)
            
            # Create columns for thumbnails
            cols = st.columns(num_cols)
            
            for i, img_path in enumerate(image_files):
                img_name = os.path.basename(img_path)
                
                # Get the completion status for the image
                completed = img_path in st.session_state.annotations
                status_icon = "✅" if completed else "❌"
                
                # Create a thumbnail with the image name and status
                with cols[i % num_cols]:
                    # Load and resize image for thumbnail
                    img = Image.open(img_path)
                    img.thumbnail(thumbnail_size)
                    
                    # Display the thumbnail with a caption
                    st.image(img, caption=f"{status_icon} {img_name}")
                    
                    # Button to select this image
                    if st.button(f"Select", key=f"select_{i}"):
                        st.session_state.selected_image = img_path
            
            # Save button
            if st.button("Save All Annotations"):
                json_path = save_annotations(st.session_state.annotations, output_file)
                st.success(f"Annotations saved to {output_file} and {json_path}")
        else:
            st.error("No images found in the directory.")
    else:
        st.error("Directory does not exist. Please enter a valid path.")

# Main content area - display selected image and annotation interface
if 'selected_image' in st.session_state and st.session_state.selected_image:
    selected_image_path = st.session_state.selected_image
    selected_image_name = os.path.basename(selected_image_path)
    
    # Display the image file name
    st.header(f"Annotating: {selected_image_name}")
    
    # Create two columns for side-by-side layout
    img_col, text_col = st.columns(2)
    
    # Display the image in the left column
    with img_col:
        img = Image.open(selected_image_path)
        st.image(img, caption=selected_image_name)
        
        # Add button to extract text with VLM if model is loaded
        if st.session_state.vlm_model_loaded:
            if st.button("Extract Text with SmolVLM"):
                with st.spinner("Extracting text from image..."):
                    extracted_text = extract_text_with_vlm(
                        selected_image_path, 
                        st.session_state.processor, 
                        st.session_state.model, 
                        st.session_state.device
                    )
                    # Store the extracted text in session state to populate the text area
                    st.session_state.current_extracted_text = extracted_text
                    # Trigger rerun to update the form
                    st.rerun()
    
    # Form for text extraction in the right column
    with text_col:
        with st.form(key='annotation_form'):
            # Create two text areas for extracted text
            st.subheader("Text Extraction")
            
            # Load existing annotations for this image if available
            existing_annotation = st.session_state.annotations.get(selected_image_path, {})
            
            # Use extracted text from VLM if available, otherwise use existing annotation
            default_model_text = ""
            if hasattr(st.session_state, 'current_extracted_text'):
                default_model_text = st.session_state.current_extracted_text
            else:
                default_model_text = existing_annotation.get("model_extracted_text", "")
            
            model_extracted_text = st.text_area(
                "Model Extracted Text (Raw from VLM)",
                value=default_model_text,
                height=150
            )
            
            corrected_text = st.text_area(
                "Corrected Text (Ground Truth)",
                value=existing_annotation.get("corrected_text", ""),
                height=150
            )
            
            # Submit button
            submit_button = st.form_submit_button(label="Save Annotation")
            
            if submit_button:
                # Save the annotation
                st.session_state.annotations[selected_image_path] = {
                    "model_extracted_text": model_extracted_text,
                    "corrected_text": corrected_text
                }
                st.success(f"Annotation saved for {selected_image_name}")
                # Save so far
                json_path = save_annotations(st.session_state.annotations, output_file)
                # st.success(f"Annotations saved to {output_file} and {json_path}")
                
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    
    # Find the current index in the list of images
    current_index = image_files.index(selected_image_path) if selected_image_path in image_files else -1
    
    # Previous button
    if current_index > 0:
        prev_image = image_files[current_index - 1]
        if col1.button("← Previous Image"):
            # Clear the current extracted text when navigating
            if hasattr(st.session_state, 'current_extracted_text'):
                del st.session_state.current_extracted_text
            st.session_state.selected_image = prev_image
            st.rerun()
    
    # Next button
    if current_index < len(image_files) - 1:
        next_image = image_files[current_index + 1]
        if col2.button("Next Image →"):
            # Clear the current extracted text when navigating
            if hasattr(st.session_state, 'current_extracted_text'):
                del st.session_state.current_extracted_text
            st.session_state.selected_image = next_image
            st.rerun()

# Display instructions if no image is selected
else:
    st.info("Select an image from the sidebar to begin annotation.")
    
    # Instructions
    st.header("Instructions")
    st.markdown("""
    1. Enter the path to your image directory in the sidebar
    2. If you want to use SmolVLM for text extraction:
       - Click "Load SmolVLM Model" in the sidebar
       - Wait for the model to load (may take a minute)
    3. Select an image thumbnail to annotate
    4. If using SmolVLM, click "Extract Text with SmolVLM" to automatically extract text
    5. Edit the extracted text if needed
    6. Enter the corrected text (ground truth)
    7. Save the annotation and move to the next image
    8. Click 'Save All Annotations' when finished to export as JSON and CSV
    """)