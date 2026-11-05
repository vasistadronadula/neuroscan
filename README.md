# NeuroScan AI

A compact Streamlit research demo for exploring brain MRI image classification and tumor-region segmentation.

> **Research and education only.** NeuroScan AI is not a medical device. Its models and image-processing heuristics have not been independently clinically validated or calibrated. Outputs are experimental model results, not diagnoses or medical advice. Do not use this project to make, delay, or rule out medical decisions; consult a qualified medical professional.

## What it does

- Classifies uploaded images with experimental brain tumor, stroke, or Alzheimer screening models.
- For the brain tumor workflow, also estimates a tumor region and displays a visualization when the model produces one.
- Accepts JPG, JPEG, PNG, and TIF uploads.
- Includes the trained model files and class-index metadata required by the app.

Model scores are raw outputs, not calibrated probabilities or diagnostic confidence. A segmentation overlay is a model visualization, not a confirmed lesion boundary. The project makes no clinical-accuracy claim.

## Quick start

### Requirements

- Windows 10 or 11
- Python 3.12 (64-bit recommended)
- The model and index files included in this repository

From the repository folder, create a clean virtual environment and install the dependencies:

```powershell
py -3.12 -m venv .venv
if (-not (Test-Path .\.venv\Scripts\python.exe)) { throw "Python 3.12 was not found or the environment could not be created." }
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`), select an analysis type, and upload an MRI image. Starting the app from the repository folder ensures its relative `models/` and `indices/` paths resolve.

## Repository contents

```text
.
├── app.py                         # Streamlit interface and inference workflow
├── indices/                       # Class-index metadata
├── models/                        # Included Keras model weights
├── requirements.txt               # Python dependencies
├── train_alzheimer_binary.py      # Experimental classifier training script
├── train_stroke_binary.py         # Experimental classifier training script
├── train_tumor_binary.py          # Experimental classifier training script
├── train_tumor_multiclass.py      # Experimental tumor-type training script
└── train_unet.py                  # Experimental tumor-segmentation training script
```

## Training data and privacy

The MRI dataset is intentionally **not included** in this public repository. Do not commit patient images, personal information, or datasets whose license does not allow redistribution. Obtain and use only data that you are authorized to process, and follow the source dataset's privacy, consent, and license requirements.

The classification training scripts are currently configured for Google Colab paths under `/content/dataset/`. The segmentation script expects matching images and masks under `dataset/brain_tumor_unet/images/` and `dataset/brain_tumor_unet/masks/`. Review each script's path and output settings before training; training can take substantial time and memory and may overwrite model files.

## Limitations

- Models, thresholds, and image heuristics are experimental and have not been validated on an independent clinical dataset.
- Model outputs can be wrong and must not be interpreted as diagnostic confidence.
- Image preprocessing may affect results; this demo is not a substitute for clinical image review.
- Always consult a qualified medical professional for diagnosis and treatment decisions.

## License

No license has been assigned to this repository. Public visibility alone does not grant permission to reuse, modify, or redistribute its code, model files, or other contents.
