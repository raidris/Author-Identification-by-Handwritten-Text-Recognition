import os
import random
from itertools import permutations
from multiprocessing import Pool, cpu_count
from typing import List, Any, Optional

import cv2
import numpy as np
import pandas as pd
import logging

from process_data import save_data

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

PROCESSED_DATA_DIR = "processed_data"  # Define this appropriately
DATA_SETS_DIR = 'data_sets'
ORG_METADATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'segments_metadata.csv')
DATASET_METADATA_PATH = os.path.join(DATA_SETS_DIR, 'dataset_metadata.csv')
MAX_SEQ_NUM = 3
MAX_DATA = 500


def resize_sequence(image_path: str, target_width: int = 224, target_height: int = 224) -> None:
    """
    Resizes the image at the specified path to the target width and target height.
    Overwrites the original image with the resized image.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Assuming sequences are grayscale images

    # Resize the image to the specified dimensions
    resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # Save the resized image back to the original path
    cv2.imwrite(image_path, resized_img)


def center_insert(base_array: np.ndarray, segment: np.ndarray, start_col: int, segment_width: int) -> np.ndarray:
    """
    Insert the segment into the base_array starting at start_col, centered within the segment_width.
    """
    # Calculate padding required on each side to center the segment
    padding = (segment_width - segment.shape[1]) // 2

    # Create a blank segment of segment_width
    blank_segment = np.full((base_array.shape[0], segment_width), 255, dtype=base_array.dtype)

    # Insert the segment into the blank segment at the centered position
    blank_segment[:, padding:padding + segment.shape[1]] = segment

    # Insert the segment into the base array starting at start_col
    base_array[:, start_col:start_col + segment_width] = blank_segment
    return base_array


def process_segments_into_sequences(segments: List[np.ndarray], max_width: int,
                                    max_seq: int = 2, cap: Optional[float] = None) -> List[np.ndarray]:
    sequences = []
    formed_combinations_hashes = set()

    # Filter out segments with width 60px or less
    segments = [segment for segment in segments if segment.shape[1] > 55]

    # If segments is empty or has fewer items than max_seq, return an empty list
    if len(segments) < max_seq:
        return []

    # Adjust max_seq to be the minimum of its current value and the length of segments
    max_seq = min(max_seq, len(segments))

    attempts = 0  # Counter to prevent infinite loop
    max_attempts = 10 * len(segments)  # Arbitrary cap on number of attempts

    while (cap is None or len(sequences) < cap) and attempts < max_attempts:
        sampled_segments = random.sample(list(enumerate(segments)), max_seq)
        indices, segments_comb = zip(*sampled_segments)

        # Create hashes for all permutations of the current combination
        permutation_hashes = set(hash(tuple(sorted(permutation))) for permutation in permutations(indices))

        # Check if any of the permutation hashes is in the formed_combinations_hashes
        if formed_combinations_hashes.intersection(permutation_hashes):
            attempts += 1
            continue

        # Record the combinations to avoid duplicates
        formed_combinations_hashes.update(permutation_hashes)

        # Build the sequence
        sequence_template = np.full((segments[0].shape[0], max_seq * max_width), 255, dtype=segments[0].dtype)
        for i, segment in enumerate(segments_comb):
            start_col = i * max_width
            sequence_template = center_insert(sequence_template, segment, start_col, max_width)

        resized_sequence = cv2.resize(sequence_template, (224, 224), interpolation=cv2.INTER_AREA)
        sequences.append(resized_sequence)

        # sequences.append(sequence_template)

    return sequences


# def process_segments_into_sequences(segments: List[np.ndarray], max_width: int) -> List[np.ndarray]:
#     sequences = []
#
#     # Filter out segments with width 60px or less
#     segments = [segment for segment in segments if segment.shape[1] > 45]
#
#     # Pre-allocated Sequence Template
#     sequence_template_proto = np.full((segments[0].shape[0], 2 * max_width), 255, dtype=segments[0].dtype)
#
#     while len(segments) > 1:  # We need at least two segments to form a sequence
#         # Randomly sample two unique segments
#         sampled_segments = random.sample(segments, 2)
#
#         # Build the sequence
#         sequence_template = sequence_template_proto.copy()
#         for i, segment in enumerate(sampled_segments):
#             start_col = i * max_width
#             sequence_template = center_insert(sequence_template, segment, start_col, max_width)
#
#         resized_sequence = cv2.resize(sequence_template, (224, 224), interpolation=cv2.INTER_AREA)
#         sequences.append(resized_sequence)
#
#         # Remove the sampled segments,so they won't be used again
#         segments = [seg for seg in segments if not any(np.array_equal(seg, sampled) for sampled in sampled_segments)]
#
#     return sequences


def process_segments_data(segments: list[dict], base_name: str, base_output_folder: str, output_dir: str,
                          max_width: int, metadata_type: str, cap: float) -> list[tuple[str, str, Any, str]]:
    sequences_metadata = []
    segments_array = []
    sequences = []

    # Read the images from the paths provided in the segments list
    for segment in segments:
        segments_array.append(cv2.imread(segment['file_path'], cv2.IMREAD_GRAYSCALE))
        sequences = process_segments_into_sequences(segments_array, max_width, MAX_SEQ_NUM, cap=cap)

    sequence_dir = os.path.join(DATA_SETS_DIR, base_output_folder, base_name)
    os.makedirs(sequence_dir, exist_ok=True)

    sequences_metadata.extend(save_data(sequences, sequence_dir, 'sequence', base_name, metadata_type))

    return sequences_metadata


def generate_data_sets(author_name: str, all_segments_metadata: list[dict],
                       max_width) -> list[tuple[str, str, Any, str]]:
    all_sequences_metadata = []
    # try:
    logging.info(f"Processing {author_name}...")

    # Create the data_sets directory and its subdirectories
    os.makedirs(DATA_SETS_DIR, exist_ok=True)

    training_segments_metadata = [item for item in all_segments_metadata
                                  if item['data_type'] == 'training' and item['author_label'] == author_name]
    validation_segments_metadata = [item for item in all_segments_metadata
                                    if item['data_type'] == 'validation' and item['author_label'] == author_name]
    test_segments_metadata = [item for item in all_segments_metadata
                              if item['data_type'] == 'test' and item['author_label'] == author_name]

    output_dir = os.path.join(DATA_SETS_DIR, author_name)

    training_sequences_metadata = process_segments_data(training_segments_metadata, author_name,
                                                        'training_set', output_dir,
                                                        max_width, 'training', MAX_DATA)
    validation_sequences_metadata = process_segments_data(validation_segments_metadata, author_name,
                                                          'validation_set', output_dir,
                                                          max_width, 'validation', MAX_DATA * 0.2)
    test_sequences_metadata = process_segments_data(test_segments_metadata, author_name,
                                                    'test_set', output_dir,
                                                    max_width, 'test', MAX_DATA * 0.2 * 0.5)

    all_sequences_metadata.extend(training_sequences_metadata)
    all_sequences_metadata.extend(validation_sequences_metadata)
    all_sequences_metadata.extend(test_sequences_metadata)

    logging.info(f"Sequencing done for {author_name}")

    # except Exception as e:
    #     logging.error(f"Error processing {author_name}: {e}")

    return all_sequences_metadata


def main() -> None:
    all_segments_metadata = pd.read_csv(ORG_METADATA_PATH).to_dict('records')

    # Find the maximum width from all segments
    max_width = max(metadata['width'] for metadata in all_segments_metadata)

    all_sequences_metadata = []

    classes = [folder for folder in os.listdir(PROCESSED_DATA_DIR) if
               os.path.isdir(os.path.join(PROCESSED_DATA_DIR, folder))]

    # Prepare the arguments for starmap
    args = [(author_name, all_segments_metadata, max_width) for author_name in classes]

    # Using multiprocessing to process images in parallel
    with Pool(cpu_count()) as pool:
        results = pool.starmap(generate_data_sets, args)

    for result in results:
        sequences_metadata = result
        all_sequences_metadata.extend(sequences_metadata)

    # Convert metadata to dataframes and save to CSV
    sequences_df = pd.DataFrame(all_sequences_metadata, columns=['file_path', 'author_label', 'width', 'data_type'])

    sequences_df.to_csv(os.path.join(DATA_SETS_DIR, 'sequences_metadata.csv'), index=False)

    logging.info("Metadata saved as .csv files.")


if __name__ == "__main__":
    main()
