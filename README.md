# FLEXseg

This repo aims to deploy recent and upcoming versions of *FLEXseg*, the first tool for segmentation of $9.4 \text{T}$ brain MRI. *FLEXseg* works out-of-the-box without any retraining.

## Description

This is the condensed code base to distribute *FLEXseg* to enable on an brain MRI. *FLEXseg* is proven to be capable of handling images at ultra-high resolution $\le 0.6 \text{mm}$ and field strengths $1.5-9.4 \text{T}$.

This model was trained using an adversarial game for flexible domain adaptation of convolutional neural networks in the context of 3D brain MRI segmentation. If you are looking for code to perform flexible training on your unseen data, please have a look at [FLEXseg_learning](https://github.com/MPI-Neuroinformatics/FLEXseg_learning).

## New Features and Updates

12/03/2024: $FLEXseg^\beta$ **is now publicly available** to allow extensive testing and ignite feedback to improve its capabilities.

## Usage

### Set Up
1. Clone this repository.
2. Create a virtual environment (i.e., with pip or conda) and install all the required packages. Installation is checked for Python 3.10. Here, we give you the minimal commands to install the required packages using conda:
  ```bash
  # Conda GPU
  conda env create -f env.yml

  # Conda CPU only
  conda env create -f env_cpu.yml
  ```
3. Contact [@jsteiglechner](https://github.com/jsteiglechner) to obtain recent model weights. Then copy them to `./model_weights/f8472e3be0f14e6d8302b6acfdc6c0bb_segmentation.pt`.

### Just Let It Run.
Once all requirements are installed, you can run *FLEXseg* on your own data by:
```bash
cd <flexseg_repo>/src
conda activate flexseg_env
python -m flexseg --input_path <input> --output_path <output> --subject <subject>
```
where:
- `<input>` should be the absolute path to an MRI to segment. It can also be an text-based file, e.g. .txt or .csv, with an path per line.
- `<subject>` subject or image ID. It will be used as name of the output_folder generated in `<output>`.
- `<output>` should be the absolute path where the output folder and its content should be saved.

## Data
All ultra-high-field data will be publicly available sonn under [UltraCortex](https://www.ultracortex.org/).

## Citation/Contact

If you find this work useful in your research, please consider citing our priliminary pre-print:
```tex
@article{}
```

If you have any questions regarding the usage of this tool, or any feedback, please raise an issue or contact us at: 
[@jsteiglechner](https://github.com/jsteiglechner) or [julius.steiglechner@tuebingen.mpg.de](mailto:julius.steiglechner@tuebingen.mpg.de)
