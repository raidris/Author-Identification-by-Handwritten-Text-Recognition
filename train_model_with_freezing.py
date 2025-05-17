import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from matplotlib import pyplot as plt
from keras.regularizers import l2

# Paths
DATA_SETS_DIR = "data_sets"
TRAIN_DIR = os.path.join(DATA_SETS_DIR, "training_set")
VALIDATION_DIR = os.path.join(DATA_SETS_DIR, "validation_set")
BEST_MODEL_PATH = 'best_model/best_model.h5'

LAYERS_PER_PHASE = 10
PHASES = 5

LEARNING_RATE = 0.0001
NUM_CLASSES = len(os.listdir(TRAIN_DIR))
L2_REG = 0.001  # L2 regularization factor

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 5


def combine_histories(*histories):
    keys = histories[0].history.keys()
    combined_history = {key: [] for key in keys}

    for history in histories:
        for key in keys:
            combined_history[key].extend(history.history[key])

    return combined_history


def plot_combined_history(history):
    # Plot training and validation accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Data Generators
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Callbacks
checkpoint = ModelCheckpoint(BEST_MODEL_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=2, verbose=1, mode='max', min_lr=0.000001)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=4, verbose=1, mode='max', restore_best_weights=True)

callbacks_list = [checkpoint,
                  reduce_lr,
                  early_stopping]

# Load the pre-trained ResNet-50v2 model without the top layers
base_model = ResNet50V2(weights='imagenet', include_top=False)

predictions = (Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=l2(L2_REG)))

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

all_histories = [
    # initial_history
]

for phase in range(PHASES):
    # Unfreeze the next set of layers
    for layer in base_model.layers[-(phase + 1) * LAYERS_PER_PHASE: -phase * LAYERS_PER_PHASE]:
        layer.trainable = True

    # Recompile the model with a potentially lower learning rate
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE * (0.1 ** (phase + 1))),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Continue training the model with the newly unfrozen layers
    history_fine_tune = model.fit(
        train_generator,
        epochs=EPOCHS,
        initial_epoch=all_histories[-1].epoch[-1] + 1,
        validation_data=validation_generator,
        callbacks=callbacks_list
    )

    all_histories.append(history_fine_tune)

# Combine all histories
combined_history = combine_histories(*all_histories)

plot_combined_history(combined_history)
