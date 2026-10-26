# AI Smart Waste Segregation Dustbin

A student prototype that uses image classification to sort waste into **organic** and **recyclable** categories. This repository contains a small web demo and a training script. It is a software prototype, not a finished physical dustbin.

## Features
- Upload a waste image through a simple Streamlit interface.
- Train an image classifier from your own labelled image folders.
- Display the predicted category and confidence.
- Includes test guidance, sample test cases, and a system design.

## Project status
This is a starter prototype. The model must be trained on a suitable, labelled dataset before predictions can be evaluated. Predictions are not guaranteed to be correct, so do not connect it to a real sorting mechanism without testing and safety controls.

## Requirements
- Python 3.10 or 3.11
- A computer with internet access for installing packages and downloading a dataset

## Setup
```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare a dataset
Create this folder structure and place labelled images inside each folder:

```text
data/
  train/
    organic/
    recyclable/
  validation/
    organic/
    recyclable/
```

Use images you have permission to use. Keep the classes balanced and make sure the labels are correct. The `organic` category can include food scraps and leaves. The `recyclable` category can include clean paper, cardboard, metal cans, and plastic bottles. Real waste streams may need more than two categories.

## Train the model
```bash
python train.py
```
The trained model is saved as `waste_model.keras`.

## Run the demo
```bash
streamlit run app.py
```
Upload an image. The app will show the predicted category and model confidence. The model file must exist first.

## Suggested next steps
1. Evaluate it on images not used for training.
2. Record accuracy and a confusion matrix.
3. Add a camera and microcontroller only after the image model is reliable.
4. For a physical prototype, use a motor driver and limit switches, and keep hands away from moving parts.
5. Add more categories only when you have enough labelled data for each.

## Repository contents
- `app.py` – image upload and prediction UI
- `train.py` – transfer-learning training script
- `requirements.txt` – Python dependencies
- `docs/architecture.md` – system design
- `tests/test_cases.md` – manual test cases
- `LICENSE` – MIT License
