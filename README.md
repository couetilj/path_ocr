# Path OCR Project

This repository is designed to assist with Optical Character Recognition (OCR) tasks for pathology-related images. The goal is to annotate a dataset of images and prepare it for fine-tuning an OCR model.

## Getting Started

### Prerequisites
Run the setup script to install the required dependencies:

```bash
./setup.sh
```

### Folder Structure
- `images/`: This folder contains the images to be annotated. Add your images here.
- `ocr_vlm.py`: The main script for annotating images and preparing the dataset.
- `batch_inference.py`: If you put a bunch of images into the folder and want to do batch inference, rather than running for each image individually within the streamlit app.
- `annotations.json`: This is where the predictions and ground truth are stored. This will be used to generate a dataset for fine-tuning the VLM.

### Usage
1. **Upload Images**: Place the images you want to annotate in the `images/` folder.
2. **Batch Inference**
    ```bash
    source path_ocr/bin/activate
    python3 batch_inference.py --image_dir images --output annotations.json
    ```
2. **Open the Streamlit app**: Use the `ocr_vlm.py` script to annotate the images. Launch the script using:
    ```bash
    ./dev.sh
    ```
    Follow the on-screen instructions to annotate the dataset.

### Next Steps
Once the dataset is annotated, we can proceed with fine-tuning the OCR model.

## Notes
- **Make sure to save your progress regularly while annotating**.
- If you encounter any issues, please document them for troubleshooting.
