import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('QtAgg')

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox,
    QVBoxLayout, QWidget, QPushButton, QHBoxLayout
)
from PyQt6.QtCore import Qt

# Matplotlib Einbau in PyQt6
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure

import torch
import torch.nn as nn
import torch.optim as optim

#matplotlib.use('TkAgg',force=True)

# ---------------------------
# Definition des Tiefen Neuronalen Netzes (LSTM)
# ---------------------------
class TemperatureNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super(TemperatureNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # Wir nehmen hier den letzten Zeitschritt
        out = self.fc(out[:, -1, :])
        return out

# ---------------------------
# Hilfsfunktionen: Datenaufbereitung
# ---------------------------
def create_sequences(data, window_size):
    """Erstellt Sequenzen aus 1D-Daten (als numpy Array)"""
    sequences = []
    targets = []
    for i in range(len(data) - window_size):
        sequences.append(data[i : i + window_size])
        targets.append(data[i + window_size])
    return np.array(sequences), np.array(targets)

# ---------------------------
# Beispiel- und Testdatengenerator
# ---------------------------
def generate_example_data(num_points=200, noise_level=0.5, filename="example_data.csv"):
    """
    Erzeugt synthetische Temperaturdaten (z. B. periodisch mit Rauschen) und speichert sie in einer CSV.
    """
    t = np.arange(num_points)
    # Beispiel: Sinusfunktion plus ein linearer Trend und Rauschkomponente
    temperatures = 10 + 5 * np.sin(2 * np.pi * t / 50) + 0.05 * t + np.random.normal(0, noise_level, size=num_points)
    df = pd.DataFrame({"Time": t, "Temperature": temperatures})
    df.to_csv(filename, index=False)
    return filename

# ---------------------------
# Plotfenster mit Matplotlib
# ---------------------------

class PlotWindow():
    def __init__(self, title="Plot"):
        super().__init__()
    
    def plot_data(self, x, y, anomalies=None, anomaly_indices=None, title=""):
        fig = plt.figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(x, y, label="Temperatur")
        if anomalies is not None and anomaly_indices is not None:
            ax.scatter(np.array(x)[anomaly_indices], anomalies, color='red', marker='x', s=100, label="Anomalie")
        ax.set_title(title)
        ax.set_xlabel("Zeit")
        ax.set_ylabel("Temperatur")
        ax.legend()
        fig.show()
# ---------------------------
# Hauptfenster der Anwendung
# ---------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Temperatur Analyse mit Tiefem Neuronalen Netz")
        self.resize(800, 600)

        # Zentrales Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Statusanzeige
        self.status = QMessageBox()

        # Das ML-Modell (wird beim Training erstellt)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        # Menü Anbindung
        self.create_menu()

        # Optional: Schaltflächen in der Oberfläche (Beispiel)
        self.info_button = QPushButton("Info / Hilfe")
        self.info_button.clicked.connect(self.show_info)
        self.layout.addWidget(self.info_button)
    
    def create_menu(self):
        menu = self.menuBar()

        # Menü Training
        train_menu = menu.addMenu("Training")
        load_train_action = train_menu.addAction("Trainingsdaten laden und trainieren")
        load_train_action.triggered.connect(self.train_model)

        # Menü Inferenz
        inference_menu = menu.addMenu("Inferenz")
        inference_action = inference_menu.addAction("Vorhersage ausführen")
        inference_action.triggered.connect(self.run_inference)

        # Menü Anomalieerkennung
        anomaly_menu = menu.addMenu("Anomalieerkennung")
        anomaly_action = anomaly_menu.addAction("Anomalien finden")
        anomaly_action.triggered.connect(self.anomaly_detection)
        
        # Menü Modell
        model_menu = menu.addMenu("Modell")
        save_model_action = model_menu.addAction("Modell speichern")
        save_model_action.triggered.connect(self.save_model)
        load_model_action = model_menu.addAction("Modell laden")
        load_model_action.triggered.connect(self.load_model)   


    def show_info(self):
        QMessageBox.information(
            self,
            "Info",
            "Dies ist ein Beispielprogramm, das ein tiefes neuronales Netz zur Analyse von Temperatur-Zeitreihen demonstriert.\n"
            "Mithilfe der Menüpunkte Training, Inferenz und Anomalieerkennung kannst du entsprechende Prozesse durchführen.\n\n"
            "Für Beispieldaten wird eine synthetische Zeitreihe erzeugt."
        )
    
    def train_model(self):
        """Lädt CSV-Daten und trainiert das Modell."""
        #options = QFileDialog.options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Trainingsdaten laden", "", "CSV Dateien (*.csv);;Alle Files (*)")
        if not fileName:
            return

        try:
            data = pd.read_csv(fileName)
            # Wir nehmen an, dass in der CSV eine Spalte "Temperature" existiert
            if "Temperature" not in data.columns:
                QMessageBox.critical(self, "Fehler", "CSV-Datei muss eine Spalte 'Temperature' enthalten.")
                return
            temperatures = data["Temperature"].values.astype(np.float32)
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Fehler beim Laden der Datei: {e}")
            return

        # Parameter
        window_size = 10  # Länge der Eingangssequenz
        sequences, targets = create_sequences(temperatures, window_size)
        sequences = torch.from_numpy(sequences.reshape(-1, window_size, 1)).to(self.device)
        targets = torch.from_numpy(targets.reshape(-1, 1)).to(self.device)

        # Initialisiere Modell
        self.model = TemperatureNet(input_size=1, hidden_size=50, num_layers=1, output_size=1).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        # Training
        epochs = 30
        self.status.information(self, "Training", "Training startet...")
        for epoch in range(1, epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            output = self.model(sequences)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")
        self.status.information(self, "Training", "Training abgeschlossen!")
    
    def run_inference(self):
        """Lädt eine kurze Zeitreihe und berechnet die Vorhersage. Es wird ein Diagramm angezeigt und die Vorhersage kann als CSV exportiert werden."""
        if self.model is None:
            QMessageBox.warning(self, "Warnung", "Bitte zuerst das Modell trainieren!")
            return

        # CSV mit Inferenz-Daten laden
        #options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Daten für Vorhersage laden", "", "CSV Dateien (*.csv);;Alle Files (*)")#, options=options)
        if not fileName:
            return

        try:
            data = pd.read_csv(fileName)
            if "Temperature" not in data.columns:
                QMessageBox.critical(self, "Fehler", "CSV-Datei muss eine Spalte 'Temperature' enthalten.")
                return
            temperatures = data["Temperature"].values.astype(np.float32)
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Fehler beim Laden der Datei: {e}")
            return

        # Wir erwarten als Eingabe eine Sequenz der Länge window_size
        window_size = 10
        if len(temperatures) < window_size:
            QMessageBox.critical(self, "Fehler", f"Die Eingabedaten müssen mindestens {window_size} Werte enthalten.")
            return

        input_seq = torch.from_numpy(temperatures[-window_size:].reshape(1, window_size, 1)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(input_seq)
        predicted_value = pred.item()

        # Erstelle eine Vorhersage-Zeitreihe (z.B. Fortsetzung der aktuellen Reihe)
        forecast_steps = 20
        forecast = []
        current_seq = temperatures[-window_size:].tolist()
        for _ in range(forecast_steps):
            seq_tensor = torch.tensor(np.array(current_seq).reshape(1, window_size, 1), dtype=torch.float32).to(self.device)
            with torch.no_grad():
                next_val = self.model(seq_tensor).item()
            forecast.append(next_val)
            current_seq.pop(0)
            current_seq.append(next_val)
        
        # Plot: Original Eingabe und Vorhersage
        total_x = list(range(len(temperatures))) + list(range(len(temperatures), len(temperatures)+forecast_steps))
        total_y = list(temperatures) + forecast
        plot_win = PlotWindow("Vorhersage")
        plot_win.plot_data(total_x, total_y, title="Temperatur-Vorhersage")
        

        # Möglichkeit zum Export der Vorhersage als CSV
        export_btn = QPushButton("Vorhersage als CSV exportieren")
        def export_csv():
            export_file, _ = QFileDialog.getSaveFileName(self, "CSV speichern", "forecast.csv", "CSV Dateien (*.csv);;Alle Files (*)")
            if export_file:
                df_export = pd.DataFrame({
                    "Time": total_x,
                    "Temperature": total_y
                })
                df_export.to_csv(export_file, index=False)
                QMessageBox.information(self, "Export", f"CSV-Datei exportiert nach: {export_file}")
        # Schalte den Button in einem einfachen Fenster ein
        export_win = QWidget()
        export_win.setWindowTitle("Export")
        layout = QHBoxLayout(export_win)
        layout.addWidget(export_btn)
        export_btn.clicked.connect(export_csv)
        export_win.show()

    def anomaly_detection(self):
        """Lädt eine CSV mit Temperatur-Zeitreihen, berechnet Vorhersagen und identifiziert Anomalien (basierend auf einem einfachen Fehlermaß)."""
        if self.model is None:
            QMessageBox.warning(self, "Warnung", "Bitte zuerst das Modell trainieren!")
            return

        #options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Daten für Anomalieerkennung laden", "", "CSV Dateien (*.csv);;Alle Files (*)")#, options=options)
        if not fileName:
            return
        try:
            data = pd.read_csv(fileName)
            if "Temperature" not in data.columns:
                QMessageBox.critical(self, "Fehler", "CSV-Datei muss eine Spalte 'Temperature' enthalten.")
                return
            temperatures = data["Temperature"].values.astype(np.float32)
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Fehler beim Laden der Datei: {e}")
            return

        window_size = 10
        if len(temperatures) < window_size + 1:
            QMessageBox.critical(self, "Fehler", f"Zu wenige Daten zur Anomalieerkennung benötigt werden mindestens {window_size+1} Werte.")
            return

        # Wir berechnen für jeden Punkt (beginnt bei window_size) den Vorhersagefehler
        errors = []
        for i in range(len(temperatures) - window_size):
            input_seq = torch.from_numpy(temperatures[i:i+window_size].reshape(1, window_size, 1)).to(self.device)
            self.model.eval()
            with torch.no_grad():
                pred = self.model(input_seq).item()
            actual = temperatures[i+window_size]
            errors.append(abs(pred - actual))
        errors = np.array(errors)

        # Schwellwert: z.B. Werte, die um mehr als 2 Standardabweichungen vom Mittelwert abweichen
        threshold = np.mean(errors) + 2 * np.std(errors)
        anomaly_indices = np.where(errors > threshold)[0] + window_size  # +window_size, da Fehler ab diesem Index definiert sind

        # Plot: Originaldaten mit hervorgehobenen Anomalien
        x = list(range(len(temperatures)))
        plot_win = PlotWindow("Anomalieerkennung")
        anomaly_values = temperatures[anomaly_indices] if anomaly_indices.size > 0 else None
        plot_win.plot_data(x, temperatures, anomalies=anomaly_values, anomaly_indices=anomaly_indices, title="Anomalien in der Temperaturreihe")


    def save_model(self):
        """Speichert das aktuelle Modell in einer Datei."""
        if self.model is None:
            QMessageBox.warning(self, "Warnung", "Es gibt kein Modell, das gespeichert werden könnte!")
            return

       # options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Modell speichern", "model.pt", "PyTorch Modelle (*.pt);;Alle Files (*)")#, options=options)
        if not fileName:
            return
        try:
            torch.save(self.model.state_dict(), fileName)
            QMessageBox.information(self, "Modell speichern", f"Modell erfolgreich gespeichert: {fileName}")
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Fehler beim Speichern des Modells: {e}")

    def load_model(self):
        """Lädt ein Modell aus einer Datei."""
        #options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Modell laden", "", "PyTorch Modelle (*.pt);;Alle Files (*)")#, options=options)
        if not fileName:
            return
        try:
            # Erzeuge ein neues Modell (dieses sollte der gespeicherten Architektur entsprechen)
            self.model = TemperatureNet(input_size=1, hidden_size=50, num_layers=1, output_size=1).to(self.device)
            self.model.load_state_dict(torch.load(fileName, map_location=self.device))
            self.model.eval()
            QMessageBox.information(self, "Modell laden", f"Modell erfolgreich geladen: {fileName}")
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Fehler beim Laden des Modells: {e}")        

# ---------------------------
# Main-Funktion
# ---------------------------
def main():
    # Optional: Beispieldaten erzeugen, falls nicht vorhanden
    if not os.path.exists("example_train.csv"):
        print("Erzeuge Beispieldaten 'example_train.csv'...")
        generate_example_data(num_points=3000, noise_level=0.7, filename="example_train.csv")
    if not os.path.exists("example_test.csv"):
        print("Erzeuge Testdaten 'example_test.csv'...")
        generate_example_data(num_points=500, noise_level=2.5, filename="example_test.csv")
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()