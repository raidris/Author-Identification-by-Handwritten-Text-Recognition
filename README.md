# Author Identification by Handwritten Text Recognition  

---

## Motivation

The task of identifying a writer based on handwriting is part of the work of forensic investigators. The Document and Handwriting Comparison Laboratory conducts various studies to improve identification capabilities and provide scientific validation for various methods. As part of such a study, several hundred people were asked to write a given text on lined pages that were distributed to them in the laboratory. The instructions were to copy and write the given text in their handwriting.

---

## Selection and Preparation of the Data Set for Work

### Database
The database belongs to the Israel Police and was provided for teaching and research purposes. It is prohibited to transfer the database itself or its processing to parties outside the course. Therefore, the database will not be included in the repository.

### Objective
The primary goal of this phase was to prepare a dataset suitable for training a neural network model to recognize handwritten text by segmenting provided images into lines and then words.

### Initial Image Processing

- **Loading Image and Metadata**: Image loaded in grayscale; `.mat` files provide line info.
- **Scaling and Adjusting Peaks**: Peaks adjusted to image scale; lines extracted using peak indices.
- **DBSCAN Filtering**: Filters out lines without significant text.
- **Final Line Processing**: Removes irrelevant edges, pads to standard height.

### Saving Processed Data

- **Loading Image and Corresponding Metadata**: The given image is loaded in grayscale mode. Accompanying .mat files, which contain information about text lines in the image, are also loaded.
- **Scaling and Adjusting Peaks**: Peaks, which represent potential lines of text, are adjusted to the original image scale. The test line is extracted using the closest peaks indices.
- **DBSCAN Filtering**: The DBSCAN clustering algorithm is employed to filter valid lines from the image. This method ensures that only lines containing significant text are considered.
- **Centered Insertion**: Segments centered in templates.
- **Saving**: Stored in corresponding directories.

### Main Execution

- **Parallel Processing**: To speed up the data processing, the script utilizes multiprocessing, allowing multiple images to be processed in parallel.
- **Metadata Saving**: Metadata for all segments and sequences, including file paths, author labels, and widths, are saved as CSV files for easy access and reference.

### Decisions & Considerations

- **Size & Number of Sub-images**: Each sub-image (segment) was resized to a standard height of 224 pixels while preserving its aspect ratio. This uniformity is essential when feeding data to a neural network.
- **Criteria for Omission**: To maintain dataset quality, we omitted images below certain dimensions or those lacking significant content. The rationale was to ensure the model wasn't fed noise or irrelevant data, which could hinder its performance.


### Previous Ideas

- **DBSCAN segments extraction**: Initially, we used DBSCAN clustering algorithm to extract black pixels clusters, which would be words. While this method worked, it produced different sized segments, which was against our goal of the uniformity of all segments.

---

## Additional Image Processing

### Objective

Process lines into individual segments and sequences for datasets suitable for training, validation, and testing.

### Segmentation Process

- **Binarization**: Lines are binarized using Otsu's thresholding.
- **Morphological Operations**: Using dilation with a horizontal structuring element, nearby black regions within the lines were merged to aid the segmentation.
- **HPP Analysis**: The HPP was computed for the dilated image, and thresholds were set to segment the individual segments based on white spaces.

### Sequencing Process

- **Segment Sequencing**: Sequences were resized to a target dimension of 224x224 pixels to ensure uniformity across all data.
- **Dataset Organization**: The data was organized into training, validation, and test sets, each containing sequences of images. The sequences were further organized based on author labels.

### Previous Ideas

- **Early Data Augmentation**: We wanted to increase the amount of data initially, but we realized that we could do that during the model training. Therefore, we discarded it in this stage.
- **Image Atomic Segmentation & Letter Sequencing**: Explored atomic segmentation (letters) and grouped them by width. This method allowed us to not discard any data. However, datasets were pruned to have duplicates.

---

## Saving the Data Set

### Objective

Organize and save sub-images for future use.

### Implementation

- **Sequential Image Storage**: Saved as `.jpg`.
- **Hierarchical Directory Structure**: Organized by image and type.

### Iterative Improvements

- Standardized sequences using padding.
- Ensured unique filenames.
- Early data splitting to avoid leakage.

### Considerations

- Chose `.jpg` for simplicity.
- Prevented data leakage by splitting on lines, not segments.

### Previous Ideas

- Randomized splitting led to leakage—later corrected.

---

## Choosing a Model & Way of Training

### Objective

Select an effective neural network and training method to distinguish handwriting authors.

### Data Preparation

- **Augmentation**: Applied transformations via Keras `ImageDataGenerator`.
- **Validation Data**: No augmentation.

### Model Selection

- **CNN Based Model**: Ideal for image tasks.
- **ResNet-v2**: Chosen for its depth and training efficiency.

### Model Structure

- **Base Model**: ResNet50V2 (pretrained on ImageNet).
- **Custom Layers**: Global average pooling → Dense (512) → Dropout → Final classification layer.

### Training Strategy

- **Layer Freezing**: Pretrained weights initially frozen.
- **Phased Unfreezing**: Gradual unfreezing and fine-tuning.

### Callbacks

- **Model Checkpoint**
- **Reduce Learning Rate**
- **Early Stopping**

### Training Process

- **Initial Training**: Frozen layers.
- **Fine-tuning**: Gradual unfreezing and learning rate adjustments.

### Visualization

- Combined history for plots.
- Plotted training/validation accuracy and loss.

---

## Results Presentation

### Learning Curves

Visual trends of training/validation loss and accuracy.

### Confusion Matrix

Post-training matrix shows misclassifications.

### Decisions & Considerations

- ResNet-50v2 chosen for balance.
- Tested other models like EfficientNetB0-B2, VGG16, custom ResNet-18-v2.

---

## Choosing the Development Strategy

### Objective

Ensure structured and documented project execution.

### Development Process

- **Iterative Approach**: Step-by-step from data to evaluation.
- **Experiment Logging**: Detailed documentation for reproducibility.

### Considerations

- Systematic folder structure to manage data and model versions.

---

## Analysis of the Final Network and its Performance

### Overview of Scripts

- **process_data.py**: Segmentation and feature extraction with multiprocessing.
- **generate_data_sets.py**: Dataset preparation by class.
- **train_model_no_freezing.py / train_model_with_freezing.py**: Model training strategies.
- **test_model.py**: Evaluation with metrics and visual results.

---

## Detailed Model Attempts

- Run `train_model.py` after data preparation via `process_data.py` and `generate_data_sets.py`.

### 5-Classes Model

- **Test Accuracy**: 222/250 (91.25%)
- **Validation Accuracy**: 438/500 (87.6%)

### 10-Classes Model

- **Test Accuracy**: 422/500 (84.4%)
- **Validation Accuracy**: 28/30 (93.3%)

---
