import pandas as pd
import numpy as np
import sqlite3
import ta
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DB_NAME = "trading_data.db"

def extract_candle_features(df):
    """
    Ekstrak fitur dari setiap candle untuk pembelajaran AI (V2 Pro Architecture)
    """
    # 1. Price Action Basics
    df['body'] = df['close'] - df['open']
    df['body_size'] = abs(df['body'])
    df['upper_shadow'] = df['high'] - df[['open','close']].max(axis=1)
    df['lower_shadow'] = df[['open','close']].min(axis=1) - df['low']
    df['total_range'] = df['high'] - df['low'] + 1e-9
    
    # Ratios for normalization
    df['body_ratio'] = df['body'] / df['total_range']
    df['upper_shadow_ratio'] = df['upper_shadow'] / df['total_range']
    df['lower_shadow_ratio'] = df['lower_shadow'] / df['total_range']
    
    # 2. Indicators (using ta library)
    df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi() / 100.0  # Normalized 0-1
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_width'] = bb.bollinger_wband()
    df['bb_hband_dist'] = (bb.bollinger_hband() - df['close']) / (df['atr'] + 1e-9)
    df['bb_lband_dist'] = (df['close'] - bb.bollinger_lband()) / (df['atr'] + 1e-9)

    # 3. Distance features (normalized by ATR)
    df['dist_ema20'] = (df['close'] - df['ema_20']) / (df['atr'] + 1e-9)
    df['dist_ema50'] = (df['close'] - df['ema_50']) / (df['atr'] + 1e-9)
    df['dist_ema200'] = (df['close'] - df['ema_200']) / (df['atr'] + 1e-9)
    
    # 4. Cyclical Time Features
    df['time_dt'] = pd.to_datetime(df['time'], unit='s')
    df['hour'] = df['time_dt'].dt.hour
    df['day_of_week'] = df['time_dt'].dt.dayofweek
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 23.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 23.0)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 6.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 6.0)

    # Drop intermediate columns
    cols_to_keep = [
        'time', 'open', 'high', 'low', 'close', 'tick_volume',
        'body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio',
        'rsi', 'atr', 'bb_width', 'bb_hband_dist', 'bb_lband_dist',
        'dist_ema20', 'dist_ema50', 'dist_ema200',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
    ]
    
    return df[cols_to_keep]

def label_candles(df, forward_window=10):
    """
    Label setiap candle: 0 (BUY), 1 (SELL), 2 (HOLD)
    Target: Menggunakan target statis Pips dari .env (default 10 pips TP, 5 pips SL).
    10 Pips XAUUSD setara pergerakan $1.00 dari harga.
    """
    close = df['close'].values
    labels = np.full(len(df), 2)  # Default: 2 (HOLD)
    
    tp_pips = float(os.getenv("SCALPING_TP_PIPS", 10))
    sl_pips = float(os.getenv("SCALPING_SL_PIPS", 5))
    
    # 1 pip XAUUSD = 0.1 decimal point (eg. 2000.00 -> 2000.10)
    tp_dist = tp_pips * 0.1
    sl_dist = sl_pips * 0.1
    
    for i in range(len(df) - forward_window):
        future_prices = close[i+1 : i+forward_window+1]
        current_price = close[i]
        
        # BUY signal (0)
        hit_tp_long = any(future_prices >= current_price + tp_dist)
        hit_sl_long = any(future_prices <= current_price - sl_dist)
        
        # SELL signal (1)
        hit_tp_short = any(future_prices <= current_price - tp_dist)
        hit_sl_short = any(future_prices >= current_price + sl_dist)
        
        if hit_tp_long and not hit_sl_long:
            labels[i] = 0
        elif hit_tp_short and not hit_sl_short:
            labels[i] = 1
            
    df['label'] = labels
    return df

def process_and_save():
    conn = sqlite3.connect(DB_NAME)
    
    # Check if rates_M15 exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rates_M15'")
    if not cursor.fetchone():
        print("Error: Table rates_M15 not found in database. Run collect_data.py first.")
        conn.close()
        return

    print("Processing features for M15 SCALPER...")
    print(f"TP Target: {os.getenv('SCALPING_TP_PIPS', 10)} pips, SL Target: {os.getenv('SCALPING_SL_PIPS', 5)} pips")
    df = pd.read_sql("SELECT * FROM rates_M15 ORDER BY time", conn)
    
    # Features & Labels
    df = extract_candle_features(df)
    df = label_candles(df)
    print(f"Label distribution: \\n{df['label'].value_counts()}")
    
    # Drop rows with NaN (due to indicators)
    df = df.dropna().reset_index(drop=True)
    
    # Save processed data
    df.to_sql("processed_m15_scalper", conn, if_exists='replace', index=False)
    print(f"Saved {len(df)} records to processed_m15_scalper")
    
    conn.close()
    print("Scalper Feature engineering complete!")

if __name__ == "__main__":
    process_and_save()
