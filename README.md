# Tool Pipeline: Approximating SPR Distance Between Phylogenetic Trees with Graph Neural Networks

This repository contains a tool based on Graph Neural Networks (GNNs) to predict the SPR (Subtree Prune-and-Regraft)
distance between pairs of phylogenetic trees.

The pipeline includes the `runPredictions.py` script, which automates the entire data management process: it downloads and
extracts the tree files from Zenodo, removes unnecessary suffixes, aligns the CSV file with the trees to be predicted, and
runs inference in a fully integrated workflow.


## Dataset
All the trees and CSV files with the SPR distances are allocated here:
https://zenodo.org/records/20476872

## Environment Setup & Dependencies

To ensure reproducibility and correct execution of the pipeline, this project uses Conda to manage the virtual environment.
You do not need to install the dependencies manually; the `environment.yml` file included in the project root takes care of
everything.

### Prerequisites
* **Conda** (Miniconda or Anaconda installed)
* **Git**

### Environment Setup

Open the terminal in the project root and execute the following commands:

```bash
# 1. Clone the Repository
git clone https://github.com/RenataCastanheira/LearningPhylogeneticTreeSimilarityWithGraphNeuralNetworks.git
cd LearningPhylogeneticTreeSimilarityWithGraphNeuralNetworks
```

```bash
# 2. Create the virtual environment 'gnn_env' from the configuration file
conda env create -f environment.yml

# 3. Activate the virtual environment 
conda activate gnn_env

# Note: 
# The configured environment already includes all the necessary scientific and Deep 
# Learning libraries, such as PyTorch, PyTorch Geometric, NetworkX, Biopython,
# Pandas and NumPy.
```

### How to use the tool pipeline for prediction
We developed owr tool centralized in `tool/runPredictions.py` that automatically resolves the CSV relative paths and handles
the data download from Zenodo.

To run  predictions on your own computer, go into the `tool` directory and execute:
```bash
cd <tool_dir>
python runPredictions.py
       --csv <your_prefiction_file>.csv
```
> **Note:** An example prediction file is available at:
>
> `scripts/prediction_file_example.csv`
>
> You can use it as a reference to create your own prediction file.

The `<tool_dir>` in the `cd` command refers to the project's root directory. All tool commands must be run from there.

 Optional arguments 
- `--model_weights`: path to the trained model weights. If omitted, the default model weights are used.
```bash
       --model_weights  <your_trained_weights>.pth
```
- `--out-predictions`: path where the final results will be saved. If omitted, the results are saved to
  `tool/scripts/FINAL_RESULTS.csv`.
```bash
       --out-predictions <your_final_results>.csv 
```
With this command, the script:
1.  Checks whether the `tool/repository/` folder exists locally. If it does not, it downloads the
   `phylogenetic_tree_dataset.zip` package from Zenodo and extracts all phylogenetic trees.
2. Automatically cleans and normalizes the tree file names.
3. Validates and filters the CSV file.
4. Loads the model with the trained weights and saves the final results to `your/final_results.csv`.



### How to use the tool pipeline for training 

In process...


