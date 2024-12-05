# DSC180Capstone - Assignment 1

## Overview
Assignment 1 represented an opportunity to get familiar with working on graph datasets.
In this case, we used New York City subway ridership data which was made available as
part of an open data challenge hosted by the New York City Metropolitan Transportation
Authority. My partner, Oren, and I were tasked with creating a graph representation of the
data, and then using that representation to answer various queries over both edges and
nodes of the graph.

## Setup
### Accessing Datasets
- Subway Station Information Dataset is in the repository, saved as MTA_Subway_Stations_updated.csv.   
- Edges Dataset can be accessed at: https://drive.google.com/drive/folders/1fV47SWGv5_AFPR_gRfvK1ra1LfSFCgOw
- Install Edges Dataset via the link above and save it as "edges.csv" in the Assignment1 folder
## Creating Environment  
- Requirements.txt contains all packages required to run the notebook
- `conda create --name assignment1 --file requirements.txt` will create the necessary conda environment
## Running Code  
- The jupyter notebook can be run sequentially, with relevant output displayed below cells as needed.
- To use the conda environment in jupyter notebook, you may need to configure jupyter using ipykernel
  ```
  conda activate assignment1
  conda install ipykernel
  ipython kernel install --user --name=assignment1
  ```
