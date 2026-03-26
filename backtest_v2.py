import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
import pandas as pd
import sqlite3
import numpy as np
from model_v2 import CandlePatternAI
from tqdm import tqdm
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DB_NAME = "trading_data.db"
SEQ_LEN = 50
INITIAL_BALANCE = 10000
LOT_SIZE = float(os.getenv("PRO_LOT_SIZE", 0.01))
CONFIDENCE_THRESHOLD = float(os.getenv("PRO_CONFIDENCE_THRESHOLD", 0.45))
PIP_VALUE = 10 

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

def simulate_backtest():
    print("Loading model and data...")
    scaler = joblib.load('scaler.gz')
    model = CandlePatternAI(input_size=len(FEATURE_COLS), seq_len=SEQ_LEN)
    model.load_state_dict(torch.load('xauusd_model_v2.pth', weights_only=True))
    model.eval()
    
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM processed_m15 ORDER BY time", conn)
    conn.close()
    
    features = df[FEATURE_COLS].values
    scaled_features = scaler.transform(features)
    
    print("Running batch inference...")
    dataset = TradingDataset(scaled_features, df['label'].values, seq_len=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    
    predictions = []
    confidences = []
    
    with torch.no_grad():
        for X, _ in tqdm(loader, desc="Inference"):
            output = model(X)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
            predictions.extend(pred.tolist())
            confidences.extend(conf.tolist())
            
    # Simulation Loop
    balance = INITIAL_BALANCE
    trades = []
    
    for i in range(len(predictions)):
        action = predictions[i]
        conf = confidences[i]
        idx = i + SEQ_LEN
        
        # Using threshold from .env
        if action < 2 and conf >= CONFIDENCE_THRESHOLD:
            actual_label = df.iloc[idx]['label']
            atr = df.iloc[idx]['atr']
            
            if action == actual_label:
                # Win (TP = 2 * ATR)
                profit = (atr * 2.0) * 100 * LOT_SIZE
                balance += profit
                trades.append(('WIN', action, profit))
            elif actual_label != 2:
                # Loss (SL = 1 * ATR)
                loss = -atr * 100 * LOT_SIZE
                balance += loss
                trades.append(('LOSS', action, loss))
    
    # Report
    print(f"Max Confidence reached: {max(confidences):.4f}")
    print(f"Average Confidence: {np.mean(confidences):.4f}")
    action_counts = pd.Series(predictions).value_counts().to_dict()
    print(f"Action counts (0:BUY, 1:SELL, 2:HOLD): {action_counts}")

    trades_df = pd.DataFrame(trades, columns=['Result', 'Action', 'PnL'])
    if not trades_df.empty:
        win_rate = (trades_df['Result'] == 'WIN').mean() * 100
        total_pnl = trades_df['PnL'].sum()
        
        pnl_series = trades_df['PnL'].cumsum()
        equity_curve = INITIAL_BALANCE + pnl_series
        max_dd = (equity_curve.cummax() - equity_curve).max()
        
        print("\n" + "="*40)
        print("ULTIMATE BACKTEST REPORT (Attention V2)")
        print("="*40)
        print(f"Initial Balance  : ${INITIAL_BALANCE}")
        print(f"Final Balance    : ${balance:.2f}")
        print(f"Total Net Profit : ${total_pnl:.2f}")
        print(f"Total Trades     : {len(trades_df)}")
        print(f"Win Rate         : {win_rate:.2f}%")
        print(f"Max Drawdown     : ${max_dd:.2f}")
        print(f"Minimum Balance  : ${equity_curve.min():.2f}")
        if equity_curve.min() < 0:
            print("WARNING: THE ACCOUNT WOULD HAVE BLOWN (BALANCE WENT NEGATIVE)!")
        print("="*40)
    else:
        print("No trades executed during the backtest.")

if __name__ == "__main__":
    simulate_backtest()
