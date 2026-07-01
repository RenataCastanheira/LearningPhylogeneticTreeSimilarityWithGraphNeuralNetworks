from torch.nn import ReLU, Sequential, Linear, Dropout

# =================================================================== #
# BLOCH 3: Output & MLP Regression Head
# =================================================================== #
def output_and_MLP_Regression(self, hidden_dim, dropout):
    """
    Final Regression based on concatenation of both embedding vectors from
    trees (v_a e v_b) passed through Fully Connected Network
    """
    self.fc = Sequential(
        Linear(hidden_dim * 2, hidden_dim),
        ReLU(),
        Dropout(dropout),
        Linear(hidden_dim, hidden_dim // 2),
        ReLU(),
        Dropout(dropout),
        Linear(hidden_dim // 2, 1),
    )
