# AI-Based Early Detection System for Hair and Scalp Disorders

**An explainable deep-learning web application for five-class scalp image classification**

Built with **PyTorch · MobileNetV2 · FastAPI · Grad-CAM · JavaScript · Tailwind CSS · SQLite**

</div>

> **Medical-use notice:** This repository describes an educational/research prototype for image classification and early awareness. It is not a medical device and must not be used to diagnose, rule out, or treat a condition. A prediction or heatmap is not a substitute for assessment by a qualified healthcare professional.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Research Gaps](#research-gaps)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Training Strategy](#model-training-strategy)
- [System Architecture](#system-architecture)
- [Explainable AI with Grad-CAM](#explainable-ai-with-grad-cam)
- [Evaluation and Results](#evaluation-and-results)
- [System Testing](#system-testing)
- [Technology Stack](#technology-stack)
- [Project Workflow](#project-workflow)
- [Installation and Setup](#installation-and-setup)
- [Example API Workflow](#example-api-workflow)
- [Suggested Repository Structure](#suggested-repository-structure)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Responsible Use and Privacy](#responsible-use-and-privacy)
- [Contributing](#contributing)
- [License](#license)

## Overview

Hair and scalp conditions such as alopecia areata, androgenetic alopecia, seborrheic dermatitis, and psoriasis can present with visible changes in the scalp or hair. People may not recognize these changes early, and access to a specialist may be delayed.

This project explores a computer-vision workflow that accepts a scalp photograph, processes it with a MobileNetV2-based image classifier, and returns a predicted category with confidence information. Grad-CAM heatmaps can also be generated to help inspect which image regions influenced the model's prediction. A web interface connects to a FastAPI backend, while SQLite and SQLAlchemy can store scan metadata and prediction records.

The current classification setup covers five labels:

1. Alopecia Areata
2. Psoriasis
3. Seborrheic Dermatitis
4. Androgenetic Alopecia
5. Normal Scalp

The project is designed for **early awareness and research**, not autonomous diagnosis. Classification into these categories does not automatically estimate severity, establish disease progression, or predict future hair loss; those tasks require separate methods and validation.

## Problem Statement

Recognizing hair and scalp disorders from photographs can be challenging. Visual signs may be subtle, image quality can vary, and several conditions may share overlapping appearances. A non-specialist may not know when a change warrants professional attention.

The project investigates whether a deep-learning model can learn visual patterns from labelled scalp photographs and provide a preliminary image category through an accessible web interface. It also explores visual explanations to make model behavior easier to inspect.

## Objectives

- Build a multi-class image-classification model for five scalp image categories.
- Standardize image validation, resizing, normalization, and label handling.
- Apply transfer learning using MobileNetV2 to support efficient inference.
- Develop a web workflow for uploading images and displaying prediction results.
- Expose model inference through a FastAPI REST API.
- Integrate Grad-CAM overlays to visualize image regions influencing predictions.
- Evaluate performance using accuracy, precision, recall, F1-score, and a confusion matrix.
- Support optional scan-history and confidence-score logging through SQLAlchemy and SQLite.
- Document limitations and the additional validation needed before any clinical use.

## Research Gaps

The project is motivated by challenges that often arise when translating image-classification research into practical systems:

### 1. Dataset standardization and representation
Image datasets may vary in capture conditions, class balance, resolution, annotation quality, and source population. These differences can cause a model to learn dataset-specific cues rather than generalizable patterns.

### 2. External and prospective validation
A good result on a random holdout from one dataset does not demonstrate performance in other clinics, locations, camera types, or populations. External and prospective evaluation would be needed to assess real-world generalizability.

### 3. Explainability
A predicted label alone provides limited information about why the model made a decision. Grad-CAM can help visualize influential image regions, but a heatmap does not prove that the model relied on medically relevant features or that the prediction is correct.

### 4. Population diversity
Performance should be investigated across skin phototypes, hair textures, hair density, ages, lighting conditions, and image-capture devices. The available dataset's demographic coverage must be documented rather than assumed.

### 5. Longitudinal monitoring and severity assessment
A single image cannot establish progression over time. Longitudinal tracking requires repeated images captured under suitable conditions, while severity estimation requires appropriately labelled severity data and a separately validated model.

### 6. Clinical utility
Clinical claims require evaluation with qualified professionals and appropriate study design. A software prototype and retrospective image classification alone do not establish clinical effectiveness.

## Key Features

- **Five-class image classification:** predicts one of the classes represented in the training dataset.
- **Web-based image upload:** JavaScript Fetch API sends an image to the backend.
- **FastAPI inference service:** separates the model API from the frontend.
- **Grad-CAM visualization:** creates an overlay for inspecting model attention.
- **Optional scan logging:** SQLAlchemy and SQLite can record metadata, timestamps, and confidence scores.
- **Modular architecture:** frontend, inference API, model, explainability, and persistence can be maintained separately.
- **Evaluation workflow:** supports quantitative metrics and confusion-matrix analysis.

Only describe scan logging, heatmap downloads, or any other feature as implemented if it exists in the current codebase.

## Dataset

The supplied dataset summary contains **3,629 scalp images across five classes**.

| Class | Number of images | Average file size (KB) | Minimum (KB) | Maximum (KB) |
|---|---:|---:|---:|---:|
| Alopecia Areata | 1,136 | 1,253.45 | 1,135.87 | 1,563.15 |
| Psoriasis | 1,134 | 1,313.05 | 1,141.63 | 1,912.69 |
| Seborrheic Dermatitis | 602 | 1,340.22 | 1,213.32 | 1,613.45 |
| Androgenetic Alopecia | 495 | 1,305.18 | 1,187.80 | 1,750.07 |
| Normal Scalp | 262 | 1,151.76 | 1,098.19 | 1,235.67 |
| **Total** | **3,629** | — | — | — |

### Dataset observations

- Alopecia Areata and Psoriasis are the largest classes, with 1,136 and 1,134 images respectively.
- Normal Scalp is the smallest class, with 262 images.
- The dataset is imbalanced, so accuracy alone may conceal poor performance on smaller classes.
- The dataset description reports a source-image resolution of **1872 × 1053 pixels**. Images are resized to the input dimensions expected by the model during preprocessing.
- File-size statistics are metadata; they do not indicate image quality or diagnostic reliability.

### Dataset provenance

Before publishing or redistributing the data, document its source, licence, permitted use, labelling process, and any consent or privacy requirements. Do not upload private or identifiable patient images without appropriate authorization. If the dataset is not licensed for redistribution, publish instructions for obtaining it instead of committing the images.

## Data Preprocessing

The pipeline prepares images for consistent model input:

1. **Load and validate:** open supported image files and handle corrupt or unreadable files.
2. **Verify labels:** check directory names and ensure each image is assigned to the correct class.
3. **Standardize image format:** convert images to the colour-channel format expected by the model.
4. **Resize:** transform images to the model's required input size.
5. **Normalize:** scale pixel values using the preprocessing expected by the selected MobileNetV2 weights.
6. **Augment training data:** where appropriate, apply modest rotation, horizontal flipping, or zooming to training images only.
7. **Split the dataset:** create training, validation, and test subsets.
8. **Prevent leakage:** where subject or patient identifiers are available, keep images from the same subject within one split. Check for duplicate or near-duplicate images across splits.

The source images are reported as 1872 × 1053 pixels; this is not necessarily the final model input size. The exact resize dimensions and normalization must match the training code.

## Model Training Strategy

### MobileNetV2 transfer learning

MobileNetV2 is a lightweight convolutional neural network often used when compute and inference resources are limited. In a typical transfer-learning workflow:

1. Load MobileNetV2 with the selected pretrained weights.
2. Replace the final classification layer with an output layer for the five classes.
3. Train the classification head and optionally fine-tune selected backbone layers.
4. Optimize the model using backpropagation on the training set.
5. Monitor validation loss and metrics to identify overfitting.
6. Save the best checkpoint using a validation criterion chosen before final testing.
7. Evaluate the selected model on an isolated test set.

### Record these experiment settings

For reproducibility, document the actual values used in the implementation:

- Pretrained weight source
- Model input dimensions and normalization
- Train/validation/test split and random seed
- Batch size and number of epochs
- Optimizer, learning rate, and loss function
- Fine-tuned layers and augmentation settings
- Checkpoint-selection criterion
- Hardware and software versions

### Dataset split

The project description specifies **80% for training and 20% reserved for validation/testing**. The exact division of that 20% must be stated from the code. For example, an 80/10/10 split is different from an 80/20 train/test split with no separate validation set. Do not tune hyperparameters against the final test set.

## System Architecture

The system uses a decoupled client-server architecture.

### 1. Client / Frontend

- HTML, JavaScript, and Tailwind CSS
- Image selection and upload
- JavaScript Fetch API for asynchronous requests
- Display of predicted class, confidence information, and returned heatmap
- User-facing error handling for unsupported or invalid images

### 2. API and Inference Backend

- FastAPI REST endpoints
- Uvicorn ASGI server
- PyTorch model loading and inference
- MobileNetV2 image classification
- Image validation and preprocessing
- Grad-CAM heatmap generation
- CORS configuration for the frontend origin

### 3. Persistence and Auditing

- SQLAlchemy ORM
- SQLite database
- Optional records containing scan timestamps, predicted class, and confidence score

Avoid storing uploaded images or other sensitive information unless necessary, authorized, and protected. Define retention and deletion policies if scan history is enabled.

### Request lifecycle

```text
User selects scalp image
          |
          v
JavaScript / Tailwind frontend
          |
          | HTTP request (Fetch API)
          v
FastAPI endpoint (Uvicorn)
          |
          v
Image validation and preprocessing
          |
          v
MobileNetV2 / PyTorch inference
          |
          +--------------------+
          |                    |
          v                    v
   Predicted class        Grad-CAM overlay
          |                    |
          +----------+---------+
                     |
                     v
          JSON response / image output
                     |
                     v
              Frontend display
                     |
                     v
          Optional SQLite audit record
```

The diagram describes the intended flow; exact response formats and database behavior should be aligned with the implementation.

## Explainable AI with Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) creates a coarse heatmap showing regions that influence a selected model output. The heatmap can be overlaid on the input image to support visual inspection.

The application description includes an output naming pattern such as `heatmap_<filename>`.

### Explainability checks

- Verify that the heatmap corresponds to the intended image and selected class.
- Check that the overlay is correctly aligned and resized.
- Inspect representative correct and incorrect predictions from all classes.
- Review whether the model appears to focus on relevant scalp regions or distracting background features.
- Record failure cases rather than showing only favorable examples.

**Important:** Grad-CAM does not eliminate the black-box problem, prove causal reasoning, or confirm that the model has identified a lesion. It is a debugging and interpretability aid, not a clinical validation method.

Performance Evaluation: Metrics, Confusion Matrix and Comparison

## Performance Evaluation

### 1. Validation Accuracy

| Metric | Score |
|---|---:|
| Validation Accuracy | 89.63% |

### 2. Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Alopecia Areata | 0.97 | 0.95 | 0.96 | 291 |
| Androgenetic Alopecia | 0.83 | 0.88 | 0.85 | 125 |
| Normal Scalp | 1.00 | 0.99 | 0.99 | 72 |
| Psoriasis | 0.87 | 0.90 | 0.88 | 258 |
| Seborrheic Dermatitis | 0.81 | 0.77 | 0.79 | 151 |
| **Accuracy** | — | — | **0.90** | **897** |
| **Macro Average** | **0.90** | **0.90** | **0.90** | **897** |
| **Weighted Average** | **0.90** | **0.90** | **0.90** | **897** |

### 3. Confusion Matrix

Rows represent actual labels, and columns represent predicted labels.

| Actual \ Predicted | Alopecia Areata | Androgenetic Alopecia | Normal Scalp | Psoriasis | Seborrheic Dermatitis |
|---|---:|---:|---:|---:|---:|
| Alopecia Areata | 275 | 11 | 0 | 3 | 2 |
| Androgenetic Alopecia | 4 | 110 | 0 | 6 | 5 |
| Normal Scalp | 1 | 0 | 71 | 0 | 0 |
| Psoriasis | 2 | 4 | 0 | 231 | 21 |
| Seborrheic Dermatitis | 1 | 8 | 0 | 25 | 117 |

### 4. Performance Analysis

| Class | Performance | Observation |
|---|---|---|
| Normal Scalp | Best (F1: 0.99) | Highest classification performance |
| Alopecia Areata | Strong (F1: 0.96) | High precision and recall |
| Psoriasis | Good (F1: 0.88) | Some confusion with Seborrheic Dermatitis |
| Androgenetic Alopecia | Moderate-to-good (F1: 0.85) | Some misclassification across other classes |
| Seborrheic Dermatitis | Needs improvement (F1: 0.79) | Lowest F1-score among the five classes |

### Key Observations

- **Best-performing class:** Normal Scalp, with an F1-score of 0.99.
- **Strong classification:** Alopecia Areata achieved an F1-score of 0.96.
- **Main challenge:** Confusion between Psoriasis and Seborrheic Dermatitis.
- The model correctly classified 231 Psoriasis samples and 117 Seborrheic Dermatitis samples.
- Improving discrimination between these visually similar conditions is a key opportunity for future model development.

### Metrics to report

- **Accuracy:** proportion of test examples classified correctly.
- **Precision:** among examples predicted as a class, the proportion that truly belongs to it.
- **Recall:** among actual examples of a class, the proportion correctly identified.
- **Macro F1-score:** unweighted average of class-level F1-scores, giving each class equal weight.
- **Weighted F1-score:** average F1-score weighted by class support.
- **Confusion matrix:** counts correct and incorrect predictions for every actual/predicted class pair.
- **Inference latency:** measured prediction time under documented hardware and timing conditions.

### Confusion matrix

Include a 5 × 5 confusion matrix with the class names on both axes. State whether rows represent actual labels and columns predicted labels. Also include class-wise precision, recall, F1-score, and support.

Because the Normal Scalp class is relatively small, pay special attention to its recall and false-positive patterns. Consider macro F1 and per-class results alongside overall accuracy.

### Latency and robustness

Measure inference latency over repeated runs and document whether the timing includes image decoding, preprocessing, model inference, Grad-CAM generation, and network overhead. Do not claim sub-second inference or out-of-memory prevention unless demonstrated on specified hardware and representative inputs.

For broader evaluation, test representative variations in lighting, focus, hair density, skin phototype, and capture device when appropriate data are available.

## System Testing

The described test plan covers:

### Prediction and technical testing
- Check API response structure and error handling for invalid uploads.
- Compare predicted labels against known ground-truth labels in the evaluation set.
- Measure inference latency using a repeatable procedure.

### Grad-CAM testing
- Confirm heatmap generation for supported inputs.
- Check overlay alignment and file output.
- Review examples with both correct and incorrect predictions for misleading activations.

### Frontend and API integration
- Verify successful image uploads and response rendering.
- Test CORS between the configured frontend origin and FastAPI backend.
- Test unsupported file types, oversized images, failed requests, and backend-unavailable scenarios.

### Data and persistence testing
- Verify logged timestamps, class labels, and confidence fields if persistence is implemented.
- Check database errors and ensure that logs do not expose unnecessary sensitive information.

These are test-plan items; report them as completed tests only when they have actually been executed.

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | HTML, JavaScript, Tailwind CSS, Fetch API |
| Backend API | Python, FastAPI, Uvicorn |
| Deep learning | PyTorch, MobileNetV2 |
| Explainability | Grad-CAM |
| Persistence | SQLAlchemy, SQLite |
| Evaluation | Accuracy, precision, recall, F1-score, confusion matrix |

Keep this list consistent with the actual code and dependency files.

## Project Workflow

1. Collect and verify labelled scalp images.
2. Inspect class distribution and image metadata.
3. Validate, resize, normalize, and split images.
4. Train and fine-tune MobileNetV2.
5. Select a checkpoint using validation results.
6. Evaluate on held-out test data and generate a confusion matrix.
7. Integrate the model into FastAPI.
8. Add Grad-CAM visualization.
9. Connect the frontend to the backend using Fetch API and configure CORS.
10. Add optional scan-history logging with SQLite and SQLAlchemy.
11. Test prediction quality, integration behavior, latency, and known failure cases.
12. Document the evaluation protocol, limitations, and intended use.

## Installation and Setup

The precise commands depend on the file layout and dependency versions in your repository. The steps below are a template—adjust paths and commands to match the project.

### Prerequisites

- Python version compatible with the installed PyTorch and project dependencies
- pip and a virtual environment
- A modern web browser
- Trained model checkpoint in the configured location
- Git

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

Replace the placeholders with your actual GitHub URL and repository folder.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the appropriate environment activation method for your machine or run the environment's Python executable directly.

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If the repository contains a `requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure the dependency file includes the versions needed by your implementation, especially PyTorch and any Grad-CAM library.

### 4. Configure model and application settings

- Place or download the trained checkpoint using a documented, trusted source.
- Confirm the model architecture, class order, input dimensions, and normalization match the training setup.
- Configure database and CORS settings as required by the code.
- Keep secrets and local paths out of version control.

### 5. Start the backend

Example only—the module path must match your FastAPI application:

```bash
uvicorn app.main:app --reload
```

If your entry point is not `app/main.py`, replace `app.main:app` with the correct import path. Use reload mode for local development, not as a production deployment recommendation.

### 6. Start the frontend

Serve the frontend using the development method appropriate to the project. Configure the backend's CORS allowlist for the exact frontend origin. Avoid using an unrestricted CORS policy in production.

### 7. Verify the workflow

- Open the frontend in a browser.
- Upload a permitted test image.
- Confirm that the API returns the expected response structure.
- Confirm the predicted label and confidence display correctly.
- Verify Grad-CAM output, if enabled.
- Check backend logs for errors.

## Example API Workflow

The exact route and field names depend on the implemented FastAPI application. If the upload endpoint accepts a multipart field named `file`, a request might look like this:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@sample_scalp.jpg"
```

This example assumes a `/predict` endpoint and a `file` form field; change them to match the actual API.

A response could contain a predicted class and confidence value, but the JSON schema must be copied from the real implementation. Do not treat a model score as a calibrated probability unless calibration has been evaluated.

## Suggested Repository Structure

Update this example to reflect the files that actually exist:

```text
hair-scalp-disorder-ai/
├── frontend/
│   ├── index.html
│   ├── assets/
│   └── js/
├── backend/
│   ├── main.py
│   ├── routes/
│   ├── services/
│   └── database/
├── models/
│   └── README.md
├── training/
├── evaluation/
├── explainability/
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

Do not commit virtual environments, local database files containing personal data, private images, secrets, or model weights unless the licence and repository size make that appropriate. Large checkpoints can be distributed through a documented release or model-storage service.

## Limitations

- Lighting, blur, camera differences, and dense hair coverage may affect predictions.
- The class distribution is imbalanced, particularly for Normal Scalp.
- A retrospective dataset from a limited source may not represent other settings.
- Skin phototype, hair texture, age, and device diversity require explicit assessment.
- Grad-CAM maps can be misleading and do not establish medical correctness.
- A single image does not establish severity or progression over time.
- Model confidence may be miscalibrated.
- Real-world clinical performance has not been established by the project description alone.

## Future Enhancements

- Evaluate on larger, more diverse datasets collected from multiple sources.
- Use patient-level splitting and check for duplicate-image leakage.
- Perform external validation and, where appropriate, prospective clinical studies.
- Investigate class imbalance with carefully validated sampling or loss strategies.
- Add calibrated confidence estimates and a clear low-confidence/uncertain pathway.
- Assess robustness across skin phototypes, hair types, lighting, and devices.
- Develop longitudinal tracking only with repeat-image protocols and suitable labels.
- Improve privacy controls, retention policies, and secure handling of scan history.
- Conduct usability studies with intended users and qualified clinical reviewers.

## Responsible Use and Privacy

- This project is for educational and research purposes.
- Do not use its output as a standalone diagnosis or treatment recommendation.
- Seek professional advice for persistent, worsening, painful, inflamed, or otherwise concerning symptoms.
- Use only datasets that you are authorized to process and share.
- Minimize collection of sensitive information, secure any retained records, and document retention and deletion practices.
- Do not claim fairness, clinical validity, or generalizability without appropriate evidence.


