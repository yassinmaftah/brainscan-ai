# BrainScan AI

Deep learning project that classifies brain MRI images into 4 classes: **glioma, meningioma, pituitary, no tumor**.

> Educational project. It is not a medical device and must not be used for diagnosis.

## Dataset

[Dataset name and source link], 7,023 MRI images in 4 classes (one folder per class).
The dataset is not included in the repository because of its size. Place it in `data/` like this:

```
data/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```

## Project structure

```
brainscan-ai/
├── notebooks/
│   ├── 01_data_loading_preprocessing.ipynb
│   ├── 02_exploration_preparation.ipynb
│   ├── 03_cnn_design.ipynb
│   ├── 04_training_evaluation.ipynb
│   └── 05_deployment.ipynb
├── models/            (model image, curves, class names; trained model is generated locally)
├── app.py             (Streamlit app)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Pipeline

1. **Loading and preprocessing:** extension check, try/except on loading, resize to 224x224 with OpenCV, shuffle, NumPy arrays.
2. **Preparation:** class balance check (largest/smallest class ratio about 1.23, so no rebalancing), label encoding (LabelEncoder + one-hot), stratified split (68% train / 12% validation / 20% test), normalization to [0, 1].
3. **Model:** 3 blocks of Conv2D (32, 64, 128 filters) + MaxPooling, then Dense(128), Dropout(0.5) and a softmax output. About 12.9 million parameters. Adam (learning rate 0.001), categorical cross-entropy.
4. **Training:** batch size 32, up to 30 epochs, ModelCheckpoint and EarlyStopping on validation loss. Best epoch: 8 (training time 12.9 minutes on CPU).
5. **Evaluation:** on the test set, used once at the end.

## Results (test set, 1,405 images)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| glioma | 0.978 | 0.941 | 0.959 |
| meningioma | 0.933 | 0.930 | 0.932 |
| notumor | 0.973 | 0.998 | 0.985 |
| pituitary | 0.975 | 0.983 | 0.979 |

Overall accuracy: **96.5%**.

- Most frequent confusion: glioma predicted as meningioma (17 images).
- 11 tumor images were predicted as "no tumor", the most serious type of error.
- 22 of the 49 errors had a confidence of 90% or more, so a high confidence does not guarantee a correct result.
- The model overfits after epoch 8 (the training accuracy reaches about 99% while validation stays around 95-96%).

## How to run

Requirements: Docker Desktop.

```bash
docker compose build

# Notebooks (JupyterLab on http://localhost:8888, token shown in the logs)
docker compose up -d jupyter
docker compose logs jupyter

# Streamlit app (http://localhost:8501), after training the model
docker compose up streamlit

# Stop everything
docker compose down
```

Run the notebooks in order (01 to 05). Notebook 04 saves the best model to `models/best_model.h5`, and notebook 05 saves `models/class_names.json`. Both files are needed by the app.

## Limitations

- Trained on one public dataset, with no clinical validation.
- The model only knows these 4 classes: any other image (even a non-MRI photo) is still assigned to one of them.
- CPU training only (no GPU configuration).

## Possible improvements

Data augmentation, a deeper model or GlobalAveragePooling to reduce the large Dense layer, class weights for meningioma, and a confidence threshold for uncertain cases.