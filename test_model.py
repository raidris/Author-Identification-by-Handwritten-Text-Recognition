import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet_v2 import preprocess_input
from sklearn.metrics import confusion_matrix

# Paths
DATA_SETS_DIR = "data_sets"
TEST_DIR = os.path.join(DATA_SETS_DIR, "test_set")
VAL_DIR = os.path.join(DATA_SETS_DIR, "validation_set")
PROCESSED_DATA_DIR = "processed_data"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_model', 'best_model.h5')

BATCH_SIZE = 32


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting normalize=True.
    """
    plt.figure(figsize=(10, 10))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.title("Normalized confusion matrix")
    else:
        plt.title('Confusion matrix, without normalization')

    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                cmap=cmap, square=True, linewidths=.5,
                cbar_kws={"shrink": .8},
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# Load the best saved model
model = load_model(MODEL_PATH)

# Prepare the test data generator
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Get predictions
test_predictions = model.predict(test_generator)
test_predicted_classes = np.argmax(test_predictions, axis=1)
test_true_classes = test_generator.classes
test_class_labels = list(test_generator.class_indices.keys())

# Display misclassified samples
test_misclassified = np.where(test_predicted_classes != test_true_classes)[0]

# Keep track of already plotted true-predicted author pairs to avoid repetition
plotted_pairs = set()

test_confusion_mtx = confusion_matrix(test_true_classes, test_predicted_classes)
plot_confusion_matrix(test_confusion_mtx, test_class_labels, normalize=True)

print(f"Total test misclassified sequences: {len(test_misclassified)}\n")

for index in test_misclassified:
    true_author = test_class_labels[test_true_classes[index]]
    predicted_author = test_class_labels[test_predicted_classes[index]]

    print(f"sequence {index} predicted as {predicted_author} but was {true_author}")

    # # If we have already plotted this true-predicted author pair, continue to next iteration
    # if (true_author, predicted_author) in plotted_pairs:
    #     continue

    # Add the true-predicted author pair to the set
    plotted_pairs.add((true_author, predicted_author))

    # Construct paths for the true and predicted test line images
    true_image_path = os.path.join(PROCESSED_DATA_DIR, true_author, "test_data", "test_line#0.jpg")
    predicted_image_path = os.path.join(PROCESSED_DATA_DIR, predicted_author, "test_data", "test_line#0.jpg")

    true_image = plt.imread(true_image_path)
    predicted_image = plt.imread(predicted_image_path)

    fig, axarr = plt.subplots(2, 1, figsize=(16, 9))

    axarr[0].imshow(true_image, cmap='gray')
    axarr[0].set_title(f"Test file: {true_author}", loc='left')
    axarr[0].axis('on')
    axarr[0].set_facecolor('white')

    axarr[1].imshow(predicted_image, cmap='gray')
    axarr[1].set_title(f"Predicted file: {predicted_author}", loc='left')
    axarr[1].axis('on')
    axarr[1].set_facecolor('white')

    plt.tight_layout()
    plt.show()

# Prepare the val data generator
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

val_loss, val_accuracy = model.evaluate(val_generator, verbose=1)
print(f"\nVal Loss: {val_loss:.4f}")
print(f"Val Accuracy: {val_accuracy:.4f}")

# Get predictions
val_predictions = model.predict(val_generator)
val_predicted_classes = np.argmax(val_predictions, axis=1)
val_true_classes = val_generator.classes
val_class_labels = list(val_generator.class_indices.keys())
val_misclassified = np.where(val_predicted_classes != val_true_classes)[0]

# Keep track of already plotted true-predicted author pairs to avoid repetition
plotted_pairs = set()

val_confusion_mtx = confusion_matrix(val_true_classes, val_predicted_classes)
plot_confusion_matrix(val_confusion_mtx, val_class_labels, normalize=True)

print(f"Total validation misclassified sequences: {len(val_misclassified)}")

for index in val_misclassified:
    true_author = val_class_labels[val_true_classes[index]]
    predicted_author = val_class_labels[val_predicted_classes[index]]

    print(f"sequence {index} predicted as {predicted_author} but was {true_author}")

    # If we have already plotted this true-predicted author pair, continue to next iteration
    if (true_author, predicted_author) in plotted_pairs:
        continue

    # Add the true-predicted author pair to the set
    plotted_pairs.add((true_author, predicted_author))

    # Construct paths for the true and predicted val line images
    true_image_path = os.path.join(PROCESSED_DATA_DIR, true_author, "validation_data", "validation_line#0.jpg")
    predicted_image_path = os.path.join(PROCESSED_DATA_DIR, predicted_author, "validation_data",
                                        "validation_line#0.jpg")

    true_image = plt.imread(true_image_path)
    predicted_image = plt.imread(predicted_image_path)

    fig, axarr = plt.subplots(2, 1, figsize=(16, 9))

    axarr[0].imshow(true_image, cmap='gray')
    axarr[0].set_title(f"validation file: {true_author}", loc='left')
    axarr[0].axis('on')
    axarr[0].set_facecolor('white')

    axarr[1].imshow(predicted_image, cmap='gray')
    axarr[1].set_title(f"Predicted file: {predicted_author}", loc='left')
    axarr[1].axis('on')
    axarr[1].set_facecolor('white')

    plt.tight_layout()
    plt.show()
