import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sqlite3
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib
from model_v2 import CandlePatternAI
from tqdm import tqdm

# Configuration
DB_NAME = "trading_data.db"
SEQ_LEN = 50
BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
PATIENCE = 15

FEATURE_COLS = [
    'body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio',
    'rsi', 'atr', 'bb_width', 'bb_hband_dist', 'bb_lband_dist',
    'dist_ema20', 'dist_ema50', 'dist_ema200',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
]

class TradingDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_data():
    print("Loading scalper data from SQLite...")
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM processed_m15_scalper ORDER BY time", conn)
    conn.close()
    
    # Check if we have data
    if len(df) == 0:
        raise ValueError("No data found in processed_m15_scalper. Run feature_engine_scalper.py first.")
    
    # Normalize features
    print("Normalizing features...")
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    scaler_path = os.getenv("SCALPING_SCALER_PATH", "scaler_scalper.gz")
    joblib.dump(scaler, scaler_path)
    
    features = []
    labels = []
    
    data_array = df[FEATURE_COLS].values
    label_array = df['label'].values
    
    for i in range(SEQ_LEN, len(df)):
        features.append(data_array[i-SEQ_LEN:i])
        labels.append(label_array[i])
        
    return np.array(features), np.array(labels)

def train():
    features, labels = load_data()
    
    # Split data (80% train, 20% val)
    split = int(0.8 * len(features))
    train_features, val_features = features[:split], features[split:]
    train_labels, val_labels = labels[:split], labels[split:]
    
    train_dataset = TradingDataset(train_features, train_labels)
    val_dataset = TradingDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CandlePatternAI(input_size=len(FEATURE_COLS), seq_len=SEQ_LEN).to(device)
    
    # Weighted CrossEntropy to handle Hold Bias
    weights = torch.FloatTensor([3.3, 3.7, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4) # Added weight decay
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print(f"Starting boost training for {EPOCHS} epochs on {device}...")
    model_path = os.getenv("SCALPING_MODEL_PATH", "xauusd_model_scalper.pth")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_features, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        # Clear line before printing metrics to fix logging bug
        print(f"\rEpoch {epoch+1}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | Val Acc {val_acc:.2f}%")
        
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}! (Improved)")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

if __name__ == "__main__":
    train()
