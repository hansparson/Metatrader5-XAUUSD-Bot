import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
import pandas as pd
import sqlite3
import numpy as np
import os
from model_v2 import CandlePatternAI
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DB_NAME = "trading_data.db"
SEQ_LEN = 50
INITIAL_BALANCE = 10000
LOT_SIZE = float(os.getenv("PRO_LOT_SIZE", 0.01))

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
    print("Loading scalper model and data...")
    scaler_path = os.getenv("SCALPING_SCALER_PATH", "scaler_scalper.gz")
    model_path = os.getenv("SCALPING_MODEL_PATH", "xauusd_model_scalper.pth")
    
    scaler = joblib.load(scaler_path)
    model = CandlePatternAI(input_size=len(FEATURE_COLS), seq_len=SEQ_LEN)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM processed_m15_scalper ORDER BY time", conn)
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
            
    # Use TP and SL from .env natively
    tp_pips = float(os.getenv("SCALPING_TP_PIPS", 10))
    sl_pips = float(os.getenv("SCALPING_SL_PIPS", 5))
    
    # 1 pip = 0.1 XAUUSD movement => TP dist = tp_pips * 0.1
    # 1 pip translates to $1 profit for 0.01 lot typically (XAUUSD standard 100oz contract tick size is 0.01).
    # Assuming $10 per point for 0.01 lot => 1 pip = $1.
    TRIAL_THRESHOLD = 0.45 
    AGGRESSIVE_LOT = 0.03
    
    balance = INITIAL_BALANCE
    trades = []
    
    for i in range(len(predictions)):
        action = predictions[i]
        conf = confidences[i]
        idx = i + SEQ_LEN
        
        if action < 2 and conf >= TRIAL_THRESHOLD:
            actual_label = df.iloc[idx]['label']
            
            if action == actual_label:
                # Win (fixed TP pips)
                # 1 Lot = 100 units. 1 pip (0.1 points) = $10 per Lot.
                # So profit = tp_pips * 10 * AGGRESSIVE_LOT
                profit = tp_pips * 10.0 * AGGRESSIVE_LOT * 10  # Note: Standard XAUUSD 10 pips = $10 on 0.01 lot. So 1 pip = $1 on 0.01 lot. So 10 pips * 10 * 0.01 = $1. Wait. Let's use 1 pip * 100 * lot size.
                profit_calc = (tp_pips * 0.1) * 100 * AGGRESSIVE_LOT
                balance += profit_calc
                trades.append(('WIN', action, profit_calc))
            elif actual_label != 2:
                # Loss (fixed SL pips)
                loss_calc = -(sl_pips * 0.1) * 100 * AGGRESSIVE_LOT
                balance += loss_calc
                trades.append(('LOSS', action, loss_calc))
    
    # Report
    print(f"Max Confidence reached: {max(confidences):.4f}")
    print(f"Average Confidence: {np.mean(confidences):.4f}")
    print(f"Confidence Threshold used: {TRIAL_THRESHOLD}")
    print(f"Fixed Target TP: {tp_pips} pips")
    print(f"Fixed Target SL: {sl_pips} pips")

    trades_df = pd.DataFrame(trades, columns=['Result', 'Action', 'PnL'])
    if not trades_df.empty:
        win_rate = (trades_df['Result'] == 'WIN').mean() * 100
        total_pnl = trades_df['PnL'].sum()
        
        pnl_series = trades_df['PnL'].cumsum()
        equity_curve = INITIAL_BALANCE + pnl_series
        max_dd = (equity_curve.cummax() - equity_curve).max()
        
        print("\n" + "="*40)
        print("NEW MODEL SCALPING PROJECTION (2-YEARS)")
        print("="*40)
        print(f"Initial Balance  : ${INITIAL_BALANCE}")
        print(f"Final Balance    : ${balance:.2f}")
        print(f"Total Net Profit : ${total_pnl:.2f}")
        print(f"Total Trades     : {len(trades_df)}")
        print(f"Win Rate         : {win_rate:.2f}%")
        print(f"Max Drawdown     : ${max_dd:.2f}")
        print(f"Profit Factor    : {abs(trades_df[trades_df['PnL'] > 0]['PnL'].sum() / trades_df[trades_df['PnL'] < 0]['PnL'].sum()):.2f}" if any(trades_df['PnL'] < 0) else "PF: Inf")
        print("="*40)
    else:
        print("No trades executed during the backtest.")

if __name__ == "__main__":
    simulate_backtest()
