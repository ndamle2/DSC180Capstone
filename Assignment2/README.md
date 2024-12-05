# DSC180Capstone - Assignment 2 (Paper Reimplementation)

## Overview
Assignment 2 was reimplementing the paper  *DE-HNN: An effective neural model for Circuit Netlist representation*, which essentially proposed Directional Equivariant Hypergraph Neu-
ral Networks (DE-HNNs) as an improved way to model directed hypergraphs, such as those
often used in representations of chip design data. More specifically, DE-HNNs address the
challenges of data size and capturing long range interactions between nodes through virtual
nodes. The assignment didn’t involve fully implementing the entire process - but instead
mostly focused on replicating the representation and predictive results.

## Setup
### Data
The data for this assignment can be pulled with the github repository and is located in the superblue18 folder.

## Creating environment
- Requirements.txt contains all packages required to run the notebook
- `conda create --name assignment2 --file requirements.txt` will create the necessary conda environment

## Running Code  
- The jupyter notebook can be run sequentially, with relevant output displayed below cells as needed.
- To use the conda environment in jupyter notebook, you may need to configure jupyter using ipykernel
  ```
  conda activate assignment1
  conda install ipykernel
  ipython kernel install --user --name=assignment1
  ```
