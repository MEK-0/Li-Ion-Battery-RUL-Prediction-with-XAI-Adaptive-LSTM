import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Attention Mechanism for LSTM.
    Calculates 'Alpha' weights to highlight which time steps (cycles)
    were most critical for the RUL prediction.
    """

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # Layers to transform LSTM output into attention scores
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states):
        # hidden_states shape: (Batch, Window_Size, Hidden_Dim)

        # 1. Score calculation (Energy)
        # energy shape: (Batch, Window, Hidden_Dim)
        energy = torch.tanh(self.attn(hidden_states))

        # 2. Calculate Attention Weights (Alpha)
        # attention shape: (Batch, Window, 1)
        attention = self.v(energy)

        # 3. Normalize weights (Sum to 1 across the window)
        alpha = F.softmax(attention, dim=1)

        # 4. Context Vector (Weighted sum of inputs)
        # context shape: (Batch, Hidden_Dim)
        context = torch.sum(hidden_states * alpha, dim=1)

        return context, alpha


class RULPredictor(nn.Module):
    """
    Hybrid Attention-LSTM Model for Battery RUL Prediction.
    Architecture: [Input] -> [LSTM] -> [Attention] -> [Dense] -> [Output]
    """

    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(RULPredictor, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM Layer
        # batch_first=True -> (Batch, Sequence, Features)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)

        # Attention Layer (The 'Explainable' part)
        self.attention = Attention(hidden_dim)

        # Fully Connected Layers (Regression Head)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor (Batch, Window_Size, 4)
               4 Features: [discharge_time, min_voltage, max_temp, ambient_temp]
        Returns:
            prediction: Estimated RUL
            alpha: Attention map for XAI visualization
        """
        # 1. LSTM Processing
        # lstm_out shape: (Batch, Window, Hidden_Dim)
        lstm_out, (hn, cn) = self.lstm(x)

        # 2. Attention Mechanism
        # Focus on important time steps
        context, alpha = self.attention(lstm_out)

        # 3. Prediction
        out = self.fc1(context)
        out = self.relu(out)
        prediction = self.fc2(out)

        return prediction, alpha


if __name__ == "__main__":
    # Sanity Check: Test model structure with dummy data
    print("Testing Model Architecture...")

    # Simulate a batch of 32 samples, window size 30, and 4 features
    dummy_input = torch.randn(32, 30, 4)

    model = RULPredictor(input_dim=4, hidden_dim=64)
    pred, weights = model(dummy_input)

    print(f"Input Shape: {dummy_input.shape}")
    print(f"Prediction Shape: {pred.shape} (Expected: [32, 1])")
    print(f"Attention Weights: {weights.shape} (Expected: [32, 30, 1])")
    print("\n✅ Model Architecture is VALID.")