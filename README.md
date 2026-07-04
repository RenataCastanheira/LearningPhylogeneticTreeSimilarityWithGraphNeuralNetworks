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

If you want to train a new model from scratch or retrain it with your own data, we developed a centralized training 
pipeline in `tool/runTraining.py`.

Unlike the prediction script, the training pipeline accepts **three separate and independent datasets** (Train, 
Validation, and Test) to ensure strict partitioning without random blending or data leakage.

To start the training process, navigate to the root project directory (<tool_dir>) and execute:

```bash

cd <tool_dir>
python runTraining.py

```
> **Reproducibility Note:** Running `python runTraining.py` directly without modifying any command-line arguments will 
> automatically train a GNN model using the exact architecture, datasets, and hyperparameter configurations presented 
> in our original paper.
>


By default, the script checks for the Zenodo dataset, validates the tree references in the default CSV files
(`scripts/train_gnn.csv`, `scripts/validation_gnn.csv`, and `scripts/test_gnn.csv`), and starts GIN model training.
All training outputs, including the model weights, are stored in the `run` directory.

**Custom Training Arguments:**

You can adjust the training hyperparameters or provide custom input files directly from the command line:

```bash
python runTraining.py \
    --train-csv <path_to_your_train>.csv \
    --val-csv   <path_to_your_validation>.csv \
    --test-csv  <path_to_your_test>.csv \
    --out-dir   <runs_my_custom_experiment> \
    --num-epochs 300 \
    --batch-size 16 \
    --patience 25 \
    --lr 1e-4
```

**Available Training Flags**
- Dataset paths:
  - `--train-csv`: Path to the strict training set CSV.
  - `--val-csv`: Path to the strict validation set CSV.
  - `--test-csv`: Path to the strict test set CSV (used for final performance evaluation).
- Output directory:
  - `--out-dir`: Directory where the best weights (`best_model.pth`), loss curves (`training_curve.csv`), and training 
  metrics summaries will be stored.
- Hyperparameters:
  - `--num-epochs`: Maximum number of training epochs (default: `300`).
  - `--batch-size`: Number of pairs processed per batch (default: `16`).
  - `--patience`: Early stopping patience (epochs to wait for validation MAE improvement before halting, default: `25`).
  - `--lr`: Learning rate for the Adam optimizer (default: `1e-4`).




