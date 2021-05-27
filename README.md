# tesser

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4793426.svg)](https://doi.org/10.5281/zenodo.4793426)

Analysis of learning and neural representations of temporal community structure.

## Installation

First, create a conda environment (if you don't have one for 
tesser yet) and activate it:

```bash
conda create -n tesser python=3.7
conda activate tesser
```

Clone the repository, change directory to it, then run:

```bash
pip install -e .
```

This should install the tesser project and all dependencies.

## Running Jupyter notebooks

To run the jupyter notebooks, first install Jupyter Lab and the kernel:

```bash
pip install jupyterlab
python -m ipykernel install --user --name tesser
```

To run the notebooks, before you run Jupyter Lab you must define the 
path to a directory with the Tesser data.  For example:

```bash
export TESSER_DIR=$HOME/Dropbox/tesser_successor/
jupyter lab &
```

In Jupyter lab, load a notebook (in `tesser/jupyter`) and make sure the 
tesser kernel is selected.

## Running scripts on TACC

You will also need brainiak and ezlaunch:

```bash
pip install brainiak
pip install ezlaunch
```

## Authors

The modeling code was developed by Rodrigo Viveros Duran, 
Demetrius Manuel Hinojosa-Rowland, Neal W Morton, Athula Pudhiyidath, 
and Ida Momennejad. Code for behavioral and neural analysis was
developed by Athula Pudhiyidath and Neal W Morton.
