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
path to a directory with the Tesser BIDS format data, and the path to
save figures to. For region of interest analyses, you must also give
the path to the directory where neural results are saved. For example:

```bash
export TESSER_BIDS=$HOME/Dropbox/work/tesser/bids
export TESSER_FIGURES=$HOME/Dropbox/tesser_successor/Figures/v2
export TESSER_RESULTS=$HOME/Dropbox/work/tesser/results
jupyter lab &
```

In Jupyter lab, load a notebook (in `tesser/jupyter`) and make sure the 
tesser kernel is selected.

## Running neural analysis scripts

Running neural analysis scripts requires additional dependencies.
If you have mpi (required for brainiak) installed, you should be 
able to just run:

```bash
pip install -e .[neural]
```

If you have problems installing brainiak or mpi4py, see the 
[brainiak website](https://brainiak.org/) for installation tips.

## Authors

The modeling code was developed by Rodrigo Viveros Duran, 
Demetrius Manuel Hinojosa-Rowland, Neal W Morton, Athula Pudhiyidath, 
and Ida Momennejad. Code for behavioral and neural analysis was
developed by Athula Pudhiyidath and Neal W Morton.
