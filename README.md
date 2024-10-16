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
For an example of how to use SwitchTFI for data analysis see **example.py**.

```bash
# Run SwitchTFI analyses with the preprocessed scRNA-seq data and a previously inferred GRN as an input
python example.py
```

## License

This project is licensed under the MIT License - see the [GNU General Public License v3.0](LICENSE) file for details.

## Contact

Paul Martini - paul.martini@fau.de

Project Link: [https://github.com/bionetslab/SwitchTFI](https://github.com/bionetslab/SwitchTFI)
