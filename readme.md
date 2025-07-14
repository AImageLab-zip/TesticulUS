<div align="left" style="display: flex; align-items: center; justify-content: space-between;">
  <h1 style="margin: 0; font-size: 2em; font-weight: bold;">
    <a href="https://federicobolelli.it/pub_files/2025iciap.pdf" style="text-decoration: none; color: inherit;">
      Enhancing Testicular Ultrasound Image Classification Through Synthetic Data and Pretraining Strategies
    </a>
  </h1>
  <img src="https://ditto.ing.unimore.it/static/testiculus/logo_w_text.png" alt="TesticulUS Dataset Logo" height="60" style="margin-left: 16px;"/>
</div>

This repository contains the official resources for the [**TesticulUS**](https://ditto.ing.unimore.it/testiculus/) dataset and the [relative paper](https://federicobolelli.it/pub_files/2025iciap.pdf), as presented at the International Conference on Image Analysis and Processing (ICIAP) 2025.


## Synthetic Dataset Access

A synthetically generated version of the dataset, created to augment the original data and facilitate further research, is available for exploration and download at the following link:

[**Explore the Synthetic TesticulUS Dataset**](https://ditto.ing.unimore.it/testiculus/)

## Generating New Images

Follow these steps to generate new synthetic images using our pretrained models:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AImageLab-zip/TesticulUS
   cd TesticulUS
   ```

2. **Download pretrained model weights:**
   - Download the model weights from [Google Drive](https://drive.google.com/file/d/1-UwCh1NuuwQXmMJ1yrNlT-08wlYmtQG3/view?usp=sharing).
   - Place the downloaded `.pt` file in your preferred directory.

3. **Generate images:**
   - Run the following command to generate images. You can set `--logging_dir=<your_path>` to specify where logs and generated images will be saved (or use the default directory):
   ```bash
   python guided-diffusion/scripts/image_sample.py \
     --in_channels 1 \
     --image_size 256 \
     --batch_size <your_batch_size> \
     --num_samples <num_of_generated_images> \
     --learn_sigma True \
     --model_path <path_to_downloaded_model.pt>
   ```
   - The generated output will be saved in the specified `logging_dir` as files named `samples*.npz`.

4. **Filter generated images (optional but recommended):**
   - To filter the generated images using our custom method, run:
   ```bash
   python guided-diffusion/filter_generation.py <path_to_samples.npz> --output_path=<output_directory>
   ```

This process allows you to generate new synthetic images and optionally filter them for higher quality using our provided scripts and models.

## Citation

If you use the TesticulUS dataset or you find this code and paper useful for your research, please cite our work:

```bibtex
@inproceedings{2025ICIAP, 
    publisher={Springer},
    venue={Rome, Italy}, 
    month={Sep}, 
    year={2025}, 
    pages={1--12}, 
    booktitle={Image Analysis and Processing – ICIAP 2025}, 
    title={{Enhancing Testicular Ultrasound Image Classification Through Synthetic Data and Pretraining Strategies}}, 
    author={Morelli, Nicola and Marchesini, Kevin and Lumetti, Luca and Santi, Daniele and Grana, Costantino and Bolelli, Federico}}
```
