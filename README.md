# SwitchTFI
This repository contains SwitchTFI python package as presented in *Identifying transcription factors driving cell differentiation*.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

## Installation

Instructions on how to install and set up the project locally.

```bash
# Clone the repository
git clone git@github.com:bionetslab/SwitchTFI.git

# Navigate to the project directory
cd SwitchTFI

# Create and activate the Conda environment from the .yml file
conda env create -f switchtfi.yml
conda activate switchtfi
```


## Usage
All relevant functions are documented with docstring comments.
For an example of how to use SwitchTFI for data analysis see **example.py**. To select an example dataset set the flag to *ery*, *beta*, or *alpha*. 

```bash
# Run SwitchTFI analyses with the preprocessed scRNA-seq data and a previously inferred GRN as an input
python example.py -d ery
```

## License

This project is licensed under the MIT License - see the [GNU General Public License v3.0](LICENSE) file for details.

## Citation

If you use **SwitchTFI** in your research or publication, please cite the corresponding preprint:

[https://doi.org/10.1101/2025.01.20.633856](https://doi.org/10.1101/2025.01.20.633856)

## Contact

Paul Martini - paul.martini@fau.de

Project Link: [https://github.com/bionetslab/SwitchTFI](https://github.com/bionetslab/SwitchTFI)
