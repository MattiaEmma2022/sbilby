# sbilby

`sbilby` is a Python package built on top of the **bilby** ecosystem, providing tools for gravitational-wave data analysis and inference, with optional integration of the LIGO Scientific Collaboration (LAL) software stack.

The package is designed to be installable via **pip**, while allowing advanced functionality when LAL is available.

---
### Usage
This package can be used to reproduce the plots and results of the RNLE study of Emma:2026 through the scripts publicly available on Zenodo 

## Installation

### Basic installation (recommended)

Install the core package from PyPI:

```bash
pip install sbilby
#####Dependencies needed to run example code from the code repository
pip install -r requirements.txt

## Optional LAL support

LAL is not available via pip. Install it using conda:

```bash
conda install -c conda-forge python-lal python-lalsimulation
pip install sbilby







