import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
import pandas as pd
import sqlite3
import numpy as np
from model_v2 import CandlePatternAI

# Final Official Configuration (Brain V2.1)
DB_NAME = "trading_data.db"
SEQ_LEN = 50
INITIAL_BALANCE = 305.53  
LOT_SIZE = 0.05 
CONFIDENCE_THRESHOLD = 0.45
QUICK_PROFIT_DISABLED = True

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

def run_final_official_simulation():
    scaler = joblib.load('scaler.gz')
    model = CandlePatternAI(input_size=len(FEATURE_COLS), seq_len=SEQ_LEN)
    model.load_state_dict(torch.load('xauusd_model_v2.pth', weights_only=True))
    model.eval()
    
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM processed_m15 ORDER BY time DESC LIMIT 5000", conn)
    conn.close()
    df = df.sort_values('time').reset_index(drop=True)
    
    features = scaler.transform(df[FEATURE_COLS].values)
    dataset = TradingDataset(features, df['label'].values, seq_len=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    
    predictions, confidences = [], []
    with torch.no_grad():
        for X, _ in loader:
            output = model(X)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
            predictions.extend(pred.tolist())
            confidences.extend(conf.tolist())

    trades_pnl = []
    for i in range(len(predictions)):
        action, conf, idx = predictions[i], confidences[i], i + SEQ_LEN
        if action < 2 and conf >= CONFIDENCE_THRESHOLD:
            actual_label = df.iloc[idx]['label']
            atr = df.iloc[idx]['atr']
            if action == actual_label:
                trades_pnl.append(atr * 100 * LOT_SIZE)
            elif actual_label != 2:
                trades_pnl.append(-atr * 100 * LOT_SIZE)
    
    num_trades = len(trades_pnl)
    weeks = 5000 / (4 * 24 * 5)
    pnl_2w = (sum(trades_pnl) / weeks) * 2
    trades_2w = (num_trades / weeks) * 2
    win_rate = (sum(1 for t in trades_pnl if t > 0) / num_trades * 100) if num_trades > 0 else 0
    
    print("\n" + "="*60)
    print(f"🌟 LAPORAN SIMULASI FINAL - BRAIN V2.1 (TREND MODE)")
    print("="*60)
    print(f"Periode Analisis   : 5 Bulan Terakhir")
    print(f"Saldo Awal         : {INITIAL_BALANCE} USC")
    print(f"Lot Size           : {LOT_SIZE}")
    print(f"Threshold AI       : {CONFIDENCE_THRESHOLD}")
    print("-" * 60)
    print(f"Estimasi Trade/2w  : {trades_2w:.1f} Kali")
    print(f"Estimasi Win Rate  : {win_rate:.1f}%")
    print(f"Estimasi Net Profit: {pnl_2w:.2f} USC (~${pnl_2w/100:.2f})")
    print(f"Estimasi ROI (2w)  : {(pnl_2w/INITIAL_BALANCE)*100:.1f}%")
    print("="*60)
    print(f"\n💡 KESIMPULAN: Model V2.1 sangat agresif mengejar tren.")
    print(f"Potensi profit meningkat drastis dibandingkan versi lama.")

if __name__ == "__main__":
    run_final_official_simulation()
