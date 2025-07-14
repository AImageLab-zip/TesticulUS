import sys, os
from pathlib import Path
import argparse
from evaluations.evaluator import Evaluator
import tensorflow.compat.v1 as tf
import numpy as np
import os

import math
import os
from PIL import Image


def main():
    """
    Main function to filter a batch of samples based on activations compared to a reference set.

    This script performs the following steps:
    1. Parses command-line arguments for the sample batch file, output path, and reference activations path.
    2. Initializes a TensorFlow session with GPU memory growth enabled.
    3. Warms up the TensorFlow model.
    4. Computes activations for the provided sample batch.
    5. Loads reference activations from a specified or default file.
    6. Filters the sample batch based on precision with respect to the reference activations.
    7. Saves the filtered and excluded samples to separate .npz files.
    8. Generates and saves a grid image of the filtered samples.

    Args:
        sample_batch (str): Path to the sample batch .npz file.
        --output_path (str, optional): Directory to save the filtered results. Defaults to the parent directory of the sample batch.
        --ref_path (str, optional): Path to the reference activations .npz file. Defaults to "static/ref_acts.npz" in the script directory.

    Raises:
        SystemExit: If the reference activations file is not found.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_batch", help="path to sample batch npz file")
    parser.add_argument("--output_path", default=None, help="output path")
    parser.add_argument("--ref_path", default=None, help="ref path")

    # Parse only known args to check existence before proceeding
    args = parser.parse_args()

    if not os.path.isfile(args.sample_batch):
        print(f"Error: sample_batch file '{args.sample_batch}' does not exist.")
        sys.exit(1)

    if args.output_path is not None and not os.path.isdir(args.output_path):
        print(f"Error: output_path '{args.output_path}' does not exist or is not a directory.")
        sys.exit(1)

    if args.ref_path is not None and not os.path.isfile(args.ref_path):
        print(f"Error: ref_path file '{args.ref_path}' does not exist.")
        sys.exit(1)

    config = tf.ConfigProto(
        allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
    )
    config.gpu_options.allow_growth = True
    filter_sample = Filter(tf.Session(config=config))

    print("warming up TensorFlow...")
    filter_sample.warmup()

    print("computing generated batch activations...")
    sample_acts = filter_sample.read_activations(args.sample_batch)
    ref_path = Path(str(__file__)).parent.joinpath("static/ref_acts.npz") if args.ref_path is None else Path(args.ref_path)
    if not ref_path.is_file():
        print("Ref activation file not found!!!")
        sys.exit(-1)
    with np.load(str(ref_path)) as ref_data:
        ref_acts = ref_data["arr_0"]
    filtered_idx = filter_sample.filter_prec(ref_acts, sample_acts[0])

    output_path = str(Path(args.sample_batch).parent ) if args.output_path is None else args.output_path + \
        "/filtered_" + str(os.path.basename(args.sample_batch))
    print(f"Saving filtered samples to {output_path}") 
    filter_sample.save_filtered_npz(args.sample_batch, output_path, filtered_idx)
    output_path = str(Path(args.sample_batch).parent ) if args.output_path is None else args.output_path  + \
        "/filtered_out_" + str(os.path.basename(args.sample_batch))
    filter_sample.save_excluded_npz(args.sample_batch, output_path, filtered_idx)

    # save_npz_grid(output_path, output_path.replace(".npz", ".png"), array_name="arr_0", grid_cols=8, pad=2)

    return

class Filter(Evaluator):
    def __init__(self, session):
        super(Filter, self).__init__(session)

    def filter_prec(self, activations_ref: np.ndarray, activations_sample: np.ndarray):
        """
        Filter the activations only if they "count for precision".
        :param activations_ref: Reference activations (real samples)
        :param activations_sample: Sample activations (generated samples)
        :return: Indices of activations of the sample batch that are inside the manifold defined by the reference batch.
        """
        radii_ref = self.manifold_estimator.manifold_radii(activations_ref)
        radii_sample = self.manifold_estimator.manifold_radii(activations_sample)

        filtered_idx = self.evaluate_precision_vectors(activations_ref, radii_ref, activations_sample, radii_sample)

        return filtered_idx

    def evaluate_precision_vectors(
        self,
        features_1: np.ndarray,
        radii_1: np.ndarray,
        features_2: np.ndarray,
        radii_2: np.ndarray,
    ) -> np.ndarray:
        """
        Return the feature vectors from features_2 (the 'batch' contained the generated samples)
        that lie inside the manifold defined by features_1 + radii_1.

        :return: A subset of indices of features_2 which are inside real manifold.
        """
        features_2_status = np.zeros([len(features_2), radii_1.shape[1]], dtype=bool)

        for begin_1 in range(0, len(features_1), self.manifold_estimator.row_batch_size):
            end_1 = begin_1 + self.manifold_estimator.row_batch_size
            batch_1 = features_1[begin_1:end_1]
            for begin_2 in range(0, len(features_2), self.manifold_estimator.col_batch_size):
                end_2 = begin_2 + self.manifold_estimator.col_batch_size
                batch_2 = features_2[begin_2:end_2]
                _, batch_2_in = self.manifold_estimator.distance_block.less_thans(
                    batch_1, radii_1[begin_1:end_1], batch_2, radii_2[begin_2:end_2]
                )
                features_2_status[begin_2:end_2] |= batch_2_in

        # Vectors that are within real manifold
        in_manifold_mask = features_2_status[:, 0]

        # return features_2[in_manifold_mask]
        return np.where(in_manifold_mask)[0]

    @staticmethod
    def save_filtered_npz(input_npz_path: str, output_npz_path: str, indices: np.ndarray, array_name: str = "arr_0"):
        """
        Saves images corresponding to `indices` to a new npz file.

        :param input_npz_path: Path to the .npz file contained all the generated data.
        :param output_npz_path: Path to save the filtered .npz.
        :param indices: Indices of images to keep.
        :param array_name: Name of the array inside the npz file.
        """

        with np.load(input_npz_path) as data:
            if array_name not in data:
                raise ValueError(f"Array '{array_name}' not found in {input_npz_path}")
            images = data[array_name]

        # Select the filtered images
        filtered_images = images[indices]

        np.savez_compressed(output_npz_path, **{array_name: filtered_images})
        print(f"Saved {len(filtered_images)} filtered samples to {output_npz_path}")

    @staticmethod
    def save_excluded_npz(input_npz_path: str, output_npz_path: str, indices: np.ndarray, array_name: str = "arr_0"):
        """
        Saves images NOT corresponding to `indices` to a new npz file.

        :param input_npz_path: Path to the .npz file contained all the generated data.
        :param output_npz_path: Path to save the excluded .npz.
        :param indices: Indices of images to exclude.
        :param array_name: Name of the array inside the npz file.
        """

        with np.load(input_npz_path) as data:
            if array_name not in data:
                raise ValueError(f"Array '{array_name}' not found in {input_npz_path}")
            images = data[array_name]
        
        # Create a mask for all indices and set the excluded ones to False
        mask = np.ones(len(images), dtype=bool)
        mask[indices] = False
        
        # Select the images not in indices
        excluded_images = images[mask]

        np.savez_compressed(output_npz_path, **{array_name: excluded_images})
        print(f"Saved {len(excluded_images)} excluded samples to {output_npz_path}")

def save_npz_grid(npz_path: str, output_path: str, array_name: str = "arr_0", grid_cols: int = 8, pad: int = 2):
    """
    Save all images from an .npz file in a single PNG grid.

    :param npz_path: path to input .npz file
    :param output_path: path to output .png file
    :param array_name: name of the array in the .npz file (default 'arr_0')
    :param grid_cols: number of columns in the image grid
    :param pad: padding (in pixels) between images
    """

    with np.load(npz_path) as data:
        if array_name not in data:
            raise ValueError(f"Array '{array_name}' not found in {npz_path}")
        images = data[array_name]

    images = np.clip(images, 0, 255).astype(np.uint8)
    if images.ndim == 3:
        images = np.stack([images] * 3, axis=-1)  # (N, H, W) → (N, H, W, 3)

    N, H, W, C = images.shape
    grid_rows = math.ceil(N / grid_cols)

    canvas_width = grid_cols * W + pad * (grid_cols - 1)
    canvas_height = grid_rows * H + pad * (grid_rows - 1)
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))

    for idx, img_array in enumerate(images):
        # 2D or 3D
        if img_array.ndim == 2:
            # (H, W) → (H, W, 3)
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.ndim == 3 and img_array.shape[2] == 1:
            # (H, W, 1) → (H, W, 3)
            img_array = np.repeat(img_array, 3, axis=2)
        elif img_array.ndim != 3 or img_array.shape[2] != 3:
            print(f"Skipping image {idx}: unexpected shape {img_array.shape}")
            continue

        if img_array.dtype != np.uint8:
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        img = Image.fromarray(img_array)
        img = img.convert("RGB")
        row = idx // grid_cols
        col = idx % grid_cols
        x = col * (W + pad)
        y = row * (H + pad)
        canvas.paste(img, (x, y))

    canvas.save(output_path)
    print(f"Saved grid image to {output_path}")


if __name__ == "__main__":
    main()
    sys.exit(0)
