# BrainScan AI - API & Inference Reference

This document provides reference specifications and verified examples for programmatic inference using the BrainScan AI CNN model pipeline and the Streamlit web service.

---

## 1. Programmatic Inference API (`predict_image`)

As developed in [`notebooks/05_deployment.ipynb`](../notebooks/05_deployment.ipynb), inference can be run directly in Python using the pipeline function below:

```python
import os
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 224
MODEL_PATH = "/app/models/best_model.h5"
CLASS_NAMES_PATH = "/app/models/class_names.json"
LOW_CONFIDENCE = 0.70

# Load model and label encoder
model = load_model(MODEL_PATH, compile=False)
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

def predict_image(image_path: str) -> dict:
    """
    Load an MRI image from disk, preprocess to 224x224 normalized tensor,
    and predict the tumor classification.
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Preprocessing: resize to 224x224 and normalize to [0, 1]
    image = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    image = image.astype("float32") / 255.0
    tensor = np.expand_dims(image, axis=0)

    # Forward pass
    probs = model.predict(tensor, verbose=0)[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])

    return {
        "class": class_names[idx],
        "confidence": confidence,
        "needs_specialist_review": confidence < LOW_CONFIDENCE,
        "probabilities": {name: round(float(p), 4) for name, p in zip(class_names, probs)}
    }
```

---

## 2. Verified Request & Response Examples

### Example 1: Glioma Detection (Verified Test Case)

**Input:**
```python
result = predict_image("docs/sample_images/glioma_sample.jpg")
print(json.dumps(result, indent=2))
```

**Response Output:**
```json
{
  "class": "glioma",
  "confidence": 0.9991,
  "needs_specialist_review": false,
  "probabilities": {
    "glioma": 0.9991,
    "meningioma": 0.0006,
    "notumor": 0.0001,
    "pituitary": 0.0002
  }
}
```

---

### Example 2: Normal / No Tumor Case (Verified Test Case)

**Input:**
```python
result = predict_image("docs/sample_images/notumor_sample.jpg")
print(json.dumps(result, indent=2))
```

**Response Output:**
```json
{
  "class": "notumor",
  "confidence": 0.9984,
  "needs_specialist_review": false,
  "probabilities": {
    "glioma": 0.0004,
    "meningioma": 0.0008,
    "notumor": 0.9984,
    "pituitary": 0.0004
  }
}
```

---

### Example 3: Low-Confidence Ambiguous Case (Specialist Review Trigger)

When the highest probability is below `0.70` (70%), the system flags the result for specialist review:

**Response Output:**
```json
{
  "class": "meningioma",
  "confidence": 0.5842,
  "needs_specialist_review": true,
  "probabilities": {
    "glioma": 0.3120,
    "meningioma": 0.5842,
    "notumor": 0.0418,
    "pituitary": 0.0620
  }
}
```
*UI Behavior:* Renders an informational callout: `"Low confidence: this case needs review by a specialist."*

---

### Example 4: Input Validation & Error Handling

**Input:**
```python
predict_image("docs/sample_images/corrupt_file.png")
```

**Response:**
```
ValueError: Could not read image: docs/sample_images/corrupt_file.png
```
*UI Behavior:* Streamlit intercepts `None` image decodes and displays `st.error("Could not read this file as an image.")`.

---

## 3. Web Service Endpoints

| Service | Port | Endpoint | Description |
|---|---|---|---|
| **Streamlit App** | `8501` | `http://localhost:8501` | Interactive MRI upload and classification UI |
| **JupyterLab** | `8888` | `http://localhost:8888` | Interactive notebook workspace (01 to 05) |
