import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Für Reproduzierbarkeit der Ergebnisse.
np.random.seed(42)
torch.manual_seed(42)

# Parameter
WINDOW_SIZE = 100
NUM_EPOCHS = 100
LEARNING_RATE = 0.01
FORECAST_STEPS = 200

# ---------------------------
# Modell 1: MLP (Fully Connected Network)
# ---------------------------
class MLPNet(nn.Module):
    def __init__(self, input_size=WINDOW_SIZE, hidden_size=50, output_size=1):
        """
        Der MLP betrachtet die gesamte Sequenz als flachen Vektor.
        """
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch, seq_len, 1) -> flache Darstellung (batch, seq_len)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ---------------------------
# Modell 2: CNN (Convolutional Neural Network)
# ---------------------------
class CNNNet(nn.Module):
    def __init__(self, input_channels=1, seq_len=WINDOW_SIZE, output_size=1):
        """
        Das CNN modelliert die Sequenz als 1D-Signal.
        """
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc = nn.Linear(seq_len * 32, output_size)
    
    def forward(self, x):
        # x: (batch, seq_len, 1) -> (batch, 1, seq_len)
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ---------------------------
# Modell 3: GRU (Gated Recurrent Unit)
# ---------------------------
class GRUNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        """
        GRU-basiertes Netzwerk für die Vorhersage der Zeitreihe.
        """
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        h0 = torch.zeros(1, x.size(0), 50).to(x.device)
        out, _ = self.gru(x, h0)
        # Benutze den letzten Zeitschritt
        out = self.fc(out[:, -1, :])
        return out

# ---------------------------
# Modell 4: LSTM
# ---------------------------
class LSTMNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        """
        LSTM-basiertes Netzwerk zur Vorhersage der Zeitreihe.
        """
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# ---------------------------
# Hilfsfunktion: Erzeuge Sequenzen aus der Zeitreihe
# ---------------------------
def create_sequences(data, window_size=WINDOW_SIZE):
    sequences = []
    targets = []
    for i in range(len(data) - window_size):
        seq = data[i:i+window_size]
        target = data[i+window_size]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# ---------------------------
# Erzeuge synthetische Zeitreihe
# ---------------------------
def generate_synthetic_data(num_points=300, noise=0.5):
    t = np.arange(num_points)
    # Beispiel: Sinusfunktion mit linearem Trend plus Rauschkomponente
    data = 10 + 5 * np.sin(2 * np.pi * t / 50) + 0.05 * t + np.random.normal(0, noise, size=num_points)
    return data

# ---------------------------
# Trainingsfunktion für ein einzelnes Modell
# ---------------------------
def train_single_model(model, optimizer, criterion, train_x, train_y, num_epochs=NUM_EPOCHS):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}")
    return model

# ---------------------------
# Vorhersage-Funktion: Iterativer Multi-Step Forecast
# ---------------------------
def forecast_future(model, input_seq, forecast_steps=FORECAST_STEPS, window_size=WINDOW_SIZE, device="cpu"):
    """
    Geht iterativ vor: zum nächsten Zeitpunkt wird die Vorhersage
    in die Eingangssequenz übernommen.
    """
    model.eval()
    forecast = []
    current_seq = input_seq.clone()  # shape: (1, window_size, 1)
    
    with torch.no_grad():
        for _ in range(forecast_steps):
            pred = model(current_seq).item()
            forecast.append(pred)
            # Aktualisiere current_seq: entferne den ersten Wert, füge pred hinzu
            current_seq_np = current_seq.cpu().numpy().flatten().tolist()
            current_seq_np.pop(0)
            current_seq_np.append(pred)
            current_seq = torch.tensor(current_seq_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
    return forecast

# ---------------------------
# Hauptprogramm
# ---------------------------
if __name__ == "__main__":
    # Gerät bestimmen
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Erzeuge die synthetische Zeitreihe und bereite die Trainingsdaten vor
    data = generate_synthetic_data(num_points=300, noise=0.5)
    sequences, targets = create_sequences(data, WINDOW_SIZE)
    
    # Konvertiere die Daten in Torch-Tensoren: Form (num_samples, seq_len, 1)
    train_x = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1).to(device)
    train_y = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1).to(device)

    # Liste der Modelltypen und zugehöriger Namen
    model_types = [
        (MLPNet(input_size=WINDOW_SIZE, hidden_size=50, output_size=1), "MLPNet"),
        (CNNNet(input_channels=1, seq_len=WINDOW_SIZE, output_size=1), "CNNNet"),
        (GRUNet(input_size=1, hidden_size=50, num_layers=1, output_size=1), "GRUNet"),
        (LSTMNet(input_size=1, hidden_size=50, num_layers=1, output_size=1), "LSTMNet")
    ]
    
    # Dictionary, um Forecasts pro Modelltyp zu speichern
    forecasts = {}

    # Trainingsschleife pro Modelltyp
    criterion = nn.MSELoss()
    for model, name in model_types:
        print(f"\nTrainiere Modell {name}:")
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train_single_model(model, optimizer, criterion, train_x, train_y, NUM_EPOCHS)
        
        # Verwende die letzte Sequenz aus den Trainingsdaten für den Forecast
        input_seq = train_x[-1].unsqueeze(0)  # Form: (1, WINDOW_SIZE, 1)
        forecast = forecast_future(model, input_seq, forecast_steps=FORECAST_STEPS, window_size=WINDOW_SIZE, device=device)
        forecasts[name] = forecast

    # ---------------------------
    # Visualisierung
    # ---------------------------
    plt.figure(figsize=(12, 7))
    t_full = np.arange(len(data))
    plt.plot(t_full, data, label="Originaldaten", color="black")

    # Zeichne Vorhersagen für jeden Modelltyp. Die Vorhersagezeitpunkte liegen direkt hinter den Trainingsdaten.
    t_forecast = np.arange(len(data), len(data) + FORECAST_STEPS)
    colors = {"MLPNet": "blue", "CNNNet": "green", "GRUNet": "orange", "LSTMNet": "red"}

    for name, forecast in forecasts.items():
        plt.plot(t_forecast, forecast, marker="o", linestyle="--", color=colors.get(name, None), label=f"Forecast {name}")

    plt.xlabel("Zeit")
    plt.ylabel("Wert")
    plt.title("Vorhersage einer Zeitreihe – Modelle: MLP, CNN, GRU, LSTM")
    plt.legend()
    plt.show()
