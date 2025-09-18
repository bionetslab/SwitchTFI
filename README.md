# SwitchTFI
This repository contains the **SwitchTFI** python package as presented in *Identifying transcription factors driving cell differentiation* ([doi](https://doi.org/10.1101/2025.01.20.633856)).

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

## Installation

**SwitchTFI** can be used in two different ways:

**1. Clone the GitHub Repository**

Preprocessed example datasets and previously inferred GRNs are available via the ``switchtfi.data`` module.
```bash
# Clone the repository
git clone git@github.com:bionetslab/SwitchTFI.git

# Navigate to the project directory
cd SwitchTFI

# Create and activate the Conda environment from the .yml file
conda env create -f switchtfi.yml
conda activate switchtfi
```

**2. Install directly from Conda**

This is the simplest way to install the package. Datasets and GRNs are not included in this installation, but a usage example is included in the repository under ``/docs/example.ipynb``.

```bash
conda install -c conda-forge -c bioconda switchtfi
```


## Usage
All relevant functions are documented with **docstrings**. For details on function parameters and available options please refer to those.

- **Repository clone:**
  
  Preprocessed scRNA-seq datasets and previously inferred GRNs can be accessed via the ``switchtfi.data`` module. An example workflow is provided in ``example.py``, please also see the comments there for additional information. To select an example dataset set the flag to *ery*, *beta*, or *alpha*. 

  ```bash
  # Run SwitchTFI analyses with the preprocessed scRNA-seq data and a previously inferred GRN as an input
  python example.py -d ery
  ```
- **Conda installation:**
  
  Datasets are not included, but a usage example is provided in the repository under ``/docs/example.ipynb``. Data preprocessing is demonstrated here as well.

## License

This project is licensed under the **GNU General Public License v3.0** - see the [LICENSE](LICENSE) file for details.

## Citation

If you use **SwitchTFI** in your research or publication, please cite the corresponding preprint:

[https://doi.org/10.1101/2025.01.20.633856](https://doi.org/10.1101/2025.01.20.633856)

## Contact

Paul Martini - paul.martini@fau.de

Project Link: [https://github.com/bionetslab/SwitchTFI](https://github.com/bionetslab/SwitchTFI)
