## Dataset
All the trees and CSV files with the SPR distances are allocated here:
https://doi.org/10.5281/zenodo.20476872

## Command Line Interface (CLI)

Our tool provides two main entry points via the command line: one for training the Graph Neural Network from scratch 
(`processTree.py`) and another for running fast inference using a pre-trained model checkpoint (`predict.py`).

---
### 1. Training the Model (`processTree.py`)

Use this script to train the Siamese Graph Isomorphism Network (GIN). It automatically splits the dataset, builds the 
graph cache, handles early stopping, and evaluates performance against baseline statistics.

#### Example Usage:
```bash
python processTree.py \
    --csv /path/to/train_gnn.csv \
    --repository /path/to/nwk_repository/ \
    --out-dir runs/spr_gnn_base \
    --num-epochs 300 \
    --batch-size 16 \
    --patience 25
```
##### Tabela de Argumentos para Treino (processTree.py)
| Argumento | Tipo | Padrão (*Default*) | Descrição                                                                                                      |
| :--- | :---: | :---: |:---------------------------------------------------------------------------------------------------------------|
| `--csv` | `str` | *Required* | Path to the CSV file containing tree pairs and their true SPR distances.                                       |
| `--repository` | `str` | *Required* | Path to the folder containing all `.nwk` (Newick) files.                                                       |
| `--out-dir` | `str` | *Required* | Directory where training curves, final metrics, and the `best_model.pth` weights will be saved.                |
| `--num-epochs` | `int` | `300` | Maximum number of training epochs.                                                                             |
| `--batch-size` | `int` | `16` | Number of tree pairs processed per batch.                                                                      |
| `--patience` | `int` | `25` | Number of epochs to wait without improvement in validation MAE before triggering *Early Stopping*.             |
| `--hidden-dim` | `int` | `128` | Hidden dimension size of the GIN layers in the network.                                                       |
| `--embed-dim` | `int` | `16` | Vector dimension size for the taxonomic node ID embedding matrix.                                   |
| `--pool` | `str` | `add` | Graph pooling aggregation strategy. Choices: `add` or `mean.                                 |
| `--log-target` | `flag` | *Disabled* | If enabled, the model trains on the log-scale target $\log(1 + \text{SPR})$ log(1+SPR) to stabilize outliers. |
| `--device` | `str` | `cuda`/`cpu` | Automatic choice of training hardware (uses GPU if available).                                   |

### 2. Running Inference / Predictions (predict.py)
Use this script to load a pre-trained model checkpoint and predict SPR distances for a set of tree pairs without running 
the training loop again. It dynamically reads the model dimensions directly from the checkpoint weights.

#### Example Usage:
```bash
python predict.py \
    --csv /path/to/test_gnn.csv \
    --repository /path/to/nwk_repository/ \
    --model_weights runs/spr_gnn_base/best_model.pth \
    --out-predictions runs/spr_gnn_base/predictions_output.csv
```

##### Tabela de Argumentos para Inferência / Teste (predict.py)
| Argumento | Tipo | Padrão (*Default*) | Descrição                                                                                                   |
| :--- | :---: | :---: |:------------------------------------------------------------------------------------------------------------|
| `--csv` | `str` | *Required* | Path to the target test CSV file with the tree pairs you want to predict.                                   |
| `--repository` | `str` | *Required* | Path to the folder containing the corresponding `.nwk` files.                                               |
| `--model_weights` | `str` | *Required* | Exact path to the weight file saved during training (ex: `runs/spr_gnn_base/best_model.pth`).               |
| `--out-predictions`| `str` | `predictions_output.csv`| Path to the new CSV file where the final table containing the `predicted_spr_distance`column will be saved. |
| `--pool` | `str` | `add` | *Attention*: Must be identical to the pooling type (`add` or `mean) used during the training phase.         |
| `--hidden-dim` | `int` | `128` | Hidden dimension of the model layers (must match the training configuration).                                        |
| `--embed-dim` | `int` | `16` | Embedding dimension of the model (must match the training configuration).                                            |
| `--device` | `str` | `cpu` | Hardware where the model will run predictions (automatically detects if a GPU is available).                                 |

    
