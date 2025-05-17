from typing import List, Optional, Tuple, Any
import os
import cv2
import numpy as np
import logging
import pandas as pd
import random

from itertools import permutations
from scipy.io import loadmat
from sklearn.cluster import DBSCAN
from multiprocessing import Pool, cpu_count

# Set up logging configuration
logging.basicConfig(
    # filename='process_data.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

BASE_DIR = 'processed_data'
SPLIT_RATIO = 0.8
MAX_HEIGHT = 224
MAX_CLASSES = 10


def process_image_into_lines(image_path: str, mat_file_path: str) -> Tuple:
    # Load the image
    image = cv2.imread(image_path, 0)  # 0 for grayscale

    # Load the .mat file
    mat = loadmat(mat_file_path)

    # Extract the peaks indices and scale factor
    peaks_indices = mat['peaks_indices'].flatten()
    scale_factor = mat['SCALE_FACTOR'].flatten()[0]
    top_test_area = mat['top_test_area'].flatten()[0]

    # Convert the peaks indices to the original image scale
    peaks_indices = (peaks_indices * scale_factor).astype(int)
    top_test_area = int(top_test_area)

    # Find the closest peaks_indices to top_test_area
    top_index = (np.abs(peaks_indices - top_test_area)).argmin()

    # Extract the test line using the closest peaks_indices
    test_lines = [image[peaks_indices[top_index]:peaks_indices[top_index + 1], :]]

    # Segment all lines
    all_lines = [image[i:j, :] for i, j in zip(peaks_indices[:-1], peaks_indices[1:])]

    # Exclude the test line to get the other lines
    other_lines = [line for idx, line in enumerate(all_lines) if idx != top_index]

    test_lines = dbscan_lines_filter(test_lines)
    other_lines = dbscan_lines_filter(other_lines)

    test_lines = [filter_edges_of_line(line) for line in test_lines]
    other_lines = [filter_edges_of_line(line) for line in other_lines]

    test_lines = [pad_line_to_standard_height(line) for line in test_lines]
    other_lines = [pad_line_to_standard_height(line) for line in other_lines]

    # Shuffle other_lines before splitting
    random.shuffle(other_lines)

    # Split other_lines into training and validation sets
    training_size = int(len(other_lines) * SPLIT_RATIO)
    training_lines = other_lines[:training_size]
    validation_lines = other_lines[training_size:]

    return training_lines, validation_lines, test_lines


def dbscan_lines_filter(lines: List[np.ndarray]) -> List[np.ndarray]:
    # Filter lines using DBSCAN
    valid_lines = []
    for line in lines:
        # Prepare data for DBSCAN
        bp_coords = np.argwhere(line < 255)  # Get coordinates of black pixels

        # Skip if there are no black pixels
        if len(bp_coords) == 0:
            continue

        # Run DBSCAN on the data
        db = DBSCAN(eps=20, min_samples=800).fit(bp_coords)

        # Check if a significant cluster was found
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # '-1' label is for noise
        if n_clusters > 0:
            valid_lines.append(line)

    return valid_lines


def pad_line_to_standard_height(line_image: np.ndarray, standard_height: int = MAX_HEIGHT) -> np.ndarray:
    """
    Pads the line image vertically to reach the standard height.
    """
    height, width = line_image.shape
    top_padding = (standard_height - height) // 2
    bottom_padding = standard_height - height - top_padding

    top_pad = np.ones((top_padding, width)) * 255
    bottom_pad = np.ones((bottom_padding, width)) * 255

    return np.vstack((top_pad, line_image, bottom_pad))


def filter_edges_of_line(line_image: np.ndarray, threshold_ratio: float = 0.1) -> Optional[np.ndarray]:
    _, binary = cv2.threshold(line_image, 128, 255, cv2.THRESH_BINARY_INV)
    hpp = np.sum(binary, axis=1)

    threshold = threshold_ratio * np.max(hpp)

    # Identify start and end of segments based on threshold
    segment_start = None
    segments = []
    for i, value in enumerate(hpp):
        if value > threshold and segment_start is None:
            segment_start = i
        elif value < threshold and segment_start is not None:
            segments.append((segment_start, i))
            segment_start = None

    # Handle the case where the last segment reaches the end of the line
    if segment_start is not None:
        segments.append((segment_start, len(hpp)))

    # Extract the segments from the original image
    extracted_segments = [line_image[start:end, :] for start, end in segments]

    # Get the segment with the largest width and return it
    if not extracted_segments:
        return None

    max_height_segment = max(extracted_segments, key=lambda x: x.shape[0])
    return max_height_segment


def save_lines(lines: List[np.ndarray], output_dir: str, line_dir: str, line_type: str) -> None:
    # Create output directory if it doesn't exist
    lines_dir = os.path.join(output_dir, f'{line_dir}')
    if not os.path.exists(lines_dir):
        os.makedirs(lines_dir)

    for i, line in enumerate(lines):
        line_path = os.path.join(lines_dir, f'{line_type}#{i}.jpg')
        cv2.imwrite(line_path, line)


def process_line_into_segments(line_image: np.ndarray) -> List[np.ndarray]:
    """
    Segments the lines using morphological operations and HPP analysis.
    """

    # Ensure the image is of uint8 type
    if line_image.dtype != np.uint8:
        line_image = cv2.convertScaleAbs(line_image)

    # Binarize the image using Otsu's thresholding
    _, binary = cv2.threshold(line_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Use dilation with a horizontal structuring element to merge nearby black regions
    segment_width = 32
    dilated_image = cv2.dilate(binary, np.ones((1, segment_width)), iterations=1)

    segments = []

    # Compute horizontal projection
    h_projection = np.sum(dilated_image, axis=0)

    # Determine the threshold for white spaces based on the mean of HPP
    threshold = 0.1 * np.mean(h_projection)

    # Segment the segments
    segment_start = None

    for i, val in enumerate(h_projection):
        if val < threshold and segment_start is None:
            continue
        if val >= threshold and segment_start is None:
            segment_start = i
        if val < threshold and segment_start is not None:
            segment_end = i
            segment = line_image[:, segment_start:segment_end]
            segments.append(segment)
            segment_start = None

    # If the last segment reaches the end
    if segment_start is not None:
        segment_end = line_image.shape[1]
        segment = line_image[:, segment_start:segment_end]
        segments.append(segment)

    return segments


def save_data(data: List[np.ndarray], save_dir: str, prefix: str, author_label: str,
              data_type: str) -> list[tuple[str, str, Any, str]]:
    metadata = []

    for i, item in enumerate(data):
        file_path = os.path.join(save_dir, f'{prefix}#{i}.jpg')
        width = item.shape[1]  # get the width of the image (segment)
        cv2.imwrite(file_path, item)
        metadata.append((file_path, author_label, width, data_type))

    return metadata


def process_lines_data(lines, base_name, base_output_folder, output_dir, metadata_type):
    segments_metadata = []

    all_segments = []
    for i, line in enumerate(lines):
        segments_dir = os.path.join(output_dir, base_output_folder, f'{metadata_type}_line#{i}_segments')
        os.makedirs(segments_dir, exist_ok=True)
        segments = process_line_into_segments(line)

        segments_metadata.extend(save_data(segments, segments_dir, 'segment', base_name, metadata_type))
        all_segments.extend(segments)

    return segments_metadata, all_segments


def process_image(image_file: str, image_dir: str = "ImagesLinesRemovedBW", mat_dir: str = "DataDarkLines",
                  output_dir_base: str = "processed_data") -> list[tuple[str, str, Any, str]]:
    all_segments_metadata = []

    try:
        logging.info(f"Processing {image_file}...")

        image_path = os.path.join(image_dir, image_file)
        if not os.path.exists(image_path):
            logging.warning(f"Image file {image_path} does not exist. Skipping...")
            return all_segments_metadata

        base_name = os.path.splitext(image_file)[0]
        mat_file_path = os.path.join(mat_dir, f'{base_name}.mat')

        if not os.path.exists(mat_file_path):
            logging.warning(f".mat file {mat_file_path} does not exist. Skipping...")
            return all_segments_metadata

        training_lines, validation_lines, test_lines = process_image_into_lines(image_path, mat_file_path)
        logging.info(f"Lines segmentation done for {image_file}")

        output_dir = os.path.join(output_dir_base, base_name)

        save_lines(training_lines, output_dir, 'training_data', 'training_line')
        save_lines(validation_lines, output_dir, 'validation_data', 'validation_line')
        save_lines(test_lines, output_dir, 'test_data', 'test_line')

        logging.info(f"Lines saved for {image_file}")

        training_segments_metadata, training_segments = process_lines_data(training_lines, base_name,
                                                                           'training_data',
                                                                           output_dir, 'training')
        validation_segments_metadata, validation_segments = process_lines_data(validation_lines, base_name,
                                                                               'validation_data', output_dir,
                                                                               'validation')
        test_segments_metadata, test_segments = process_lines_data(test_lines, base_name, 'test_data',
                                                                   output_dir, 'test')

        all_segments_metadata.extend(training_segments_metadata)
        all_segments_metadata.extend(validation_segments_metadata)
        all_segments_metadata.extend(test_segments_metadata)

    except Exception as e:
        logging.error(f"Error processing {image_file}: {e}")

    return all_segments_metadata


def main() -> None:
    image_dir = "ImagesLinesRemovedBW"
    image_files = sorted(os.listdir(image_dir))[:MAX_CLASSES]  # Only take the first MAX_CLASSES images

    all_segments_metadata = []

    # Using multiprocessing to process images in parallel
    with Pool(cpu_count()) as pool:
        results = pool.map(process_image, image_files)

    for result in results:
        segments_metadata = result
        all_segments_metadata.extend(segments_metadata)

    # Convert metadata to dataframes and save to CSV
    segments_df = pd.DataFrame(all_segments_metadata, columns=['file_path', 'author_label', 'width', 'data_type'])

    segments_df.to_csv(os.path.join(BASE_DIR, 'segments_metadata.csv'), index=False)

    logging.info("Metadata saved as .csv files.")


if __name__ == "__main__":
    main()
