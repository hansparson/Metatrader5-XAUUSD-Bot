import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
import pandas as pd
import sqlite3
import numpy as np
from model_v2 import CandlePatternAI

# Global Configuration
DB_NAME = "trading_data.db"
SEQ_LEN = 50
INITIAL_BALANCE = 305.53  
LOT_SIZE = 0.10 
QUICK_PROFIT_USC = 1.0

FEATURE_COLS = [
    'body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio',
    'rsi', 'atr', 'bb_width', 'bb_hband_dist', 'bb_lband_dist',
    'dist_ema20', 'dist_ema50', 'dist_ema200',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
]

class TradingDataset(Dataset):
    def __init__(self, data, labels, seq_len=50):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        return self.data[idx : idx + self.seq_len], self.labels[idx + self.seq_len]

def run_threshold_optimization():
    scaler = joblib.load('scaler.gz')
    model = CandlePatternAI(input_size=len(FEATURE_COLS), seq_len=SEQ_LEN)
    model.load_state_dict(torch.load('xauusd_model_v2.pth', weights_only=True))
    model.eval()
    
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM processed_m15 ORDER BY time DESC LIMIT 5000", conn)
    conn.close()
    df = df.sort_values('time').reset_index(drop=True)
    
    features = df[FEATURE_COLS].values
    scaled_features = scaler.transform(features)
    
    dataset = TradingDataset(scaled_features, df['label'].values, seq_len=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    
    predictions, confidences = [], []
    with torch.no_grad():
        for X, _ in loader:
            output = model(X)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
            predictions.extend(pred.tolist())
            confidences.extend(conf.tolist())
            
    print(f"\n🔍 Max Confidence Reached: {max(confidences):.2%}")
    print(f"🔍 Avg Confidence: {np.mean(confidences):.2%}\n")

    thresholds = [0.45, 0.50, 0.55, 0.60]
    print("="*60)
    print(f"{'Threshold':<12} | {'Trades':<8} | {'WinRate':<8} | {'Net Profit (2w)':<15}")
    print("-" * 60)

    for th in thresholds:
        trades = []
        for i in range(len(predictions)):
            action, conf, idx = predictions[i], confidences[i], i + SEQ_LEN
            if action < 2 and conf >= th:
                actual_label = df.iloc[idx]['label']
                if action == actual_label:
                    trades.append(QUICK_PROFIT_USC)
                elif actual_label != 2:
                    atr = df.iloc[idx]['atr']
                    loss = -atr * 100 * LOT_SIZE 
                    trades.append(loss)
        
        num_trades = len(trades)
        win_rate = (sum(1 for t in trades if t > 0) / num_trades * 100) if num_trades > 0 else 0
        net_pnl = sum(trades)
        
        # Scale to 2 weeks
        weeks = 5000 / (4 * 24 * 5)
        pnl_2w = (net_pnl / weeks) * 2
        trades_2w = (num_trades / weeks) * 2
        
        print(f"{th:<12.2f} | {trades_2w:<8.1f} | {win_rate:<7.1f}% | {pnl_2w:<15.2f} USC")
    print("="*60)

if __name__ == "__main__":
    run_threshold_optimization()
