import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import time
import ta
import joblib
import os
import sqlite3
import warnings
from dotenv import load_dotenv
from model_v2 import CandlePatternAI

# Silence warnings for a clean production interface
warnings.filterwarnings("ignore")
# Load .env with override=True to ensure manual changes are picked up on restart
load_dotenv(override=True)

# =================================================================
# CONFIGURATION (LOADED FROM .ENV WITH DEFAULTS)
# =================================================================
TRADING_MODE = os.getenv("TRADING_MODE", "DEMO").upper()
SYMBOL_BASE = os.getenv("PRO_SYMBOL", "XAUUSD.vxc")
TIMEFRAME_M15 = mt5.TIMEFRAME_M15
SEQ_LEN = int(os.getenv("PRO_SEQ_LEN", 50))
MODEL_PATH = os.getenv("PRO_MODEL_PATH", "xauusd_model_v2.pth")
SCALER_PATH = os.getenv("PRO_SCALER_PATH", "scaler.gz")
DB_NAME = os.getenv("PRO_DB_NAME", "trading_data.db")
MAGIC_NUMBER = int(os.getenv("PRO_MAGIC_NUMBER", 20240326))
MAX_OPEN_POSITIONS = int(os.getenv("PRO_MAX_POSITIONS", 5))
PROFIT_THRESHOLD = float(os.getenv("PRO_QUICK_PROFIT", 2.0))
CONFIDENCE_THRESHOLD = float(os.getenv("PRO_CONFIDENCE_THRESHOLD", 0.60))
LOT_SIZE = float(os.getenv("PRO_LOT_SIZE", 1.0))

# Global Symbol (May be updated by auto-detection)
SYMBOL = SYMBOL_BASE

FEATURE_COLS = [
    'body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio',
    'rsi', 'atr', 'bb_width', 'bb_hband_dist', 'bb_lband_dist',
    'dist_ema20', 'dist_ema50', 'dist_ema200',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
]

# =================================================================
# ENGINE CORE FUNCTIONS
# =================================================================

def init_mt5():
    """Initializes connection to MT5 based on TRADING_MODE (DEMO/LIVE)."""
    if not mt5.initialize():
        print(f"[{datetime.now()}] MT5 Initialize failed: {mt5.last_error()}")
        return False
    
    # Load Credentials based on Mode
    prefix = "MT5_LIVE_" if TRADING_MODE == "LIVE" else "MT5_DEMO_"
    login = os.getenv(prefix + "LOGIN")
    password = os.getenv(prefix + "PASSWORD")
    server = os.getenv(prefix + "SERVER")
    
    if login and password and server:
        # Hide password in logs for security
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to {TRADING_MODE} Account: {login}...")
        if not mt5.login(int(login), password, server):
            print(f"[{datetime.now()}] Login failed for {TRADING_MODE} account {login}")
            return False
    else:
        print(f"[{datetime.now()}] Warning: No {TRADING_MODE} credentials found in .env. Using current terminal session.")
            
    # Auto-detect Correct Symbol Name (Fuzzy Search)
    global SYMBOL
    all_symbols = [s.name for s in mt5.symbols_get()]
    
    # Priority 1: Exact match from .env
    if SYMBOL_BASE in all_symbols and mt5.symbol_select(SYMBOL_BASE, True):
        SYMBOL = SYMBOL_BASE
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Symbol Syncing OK | Symbol: {SYMBOL}")
        return True
        
    # Priority 2: Fuzzy match for XAUUSD patterns
    for s in all_symbols:
        if "XAU" in s and "USD" in s:
            if mt5.symbol_select(s, True):
                SYMBOL = s
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Auto-Detected Symbol: {SYMBOL}")
                return True
                
    # Priority 3: Common candidates
    candidates = ["XAUUSD.vxc", "XAUUSD.vx", "GOLD", "XAUUSD.m", "XAUUSD+", "XAUUSD.pro"]
    for s in candidates:
        if s in all_symbols and mt5.symbol_select(s, True):
            SYMBOL = s
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Symbol Syncing OK | Symbol: {SYMBOL}")
            return True
            
    print(f"[{datetime.now()}] Error: Could not find a valid Gold symbol in MT5.")
    return False

def get_latest_data(n=300):
    """Fetches data from MT5 with a bootstrap fallback from SQLite."""
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME_M15, 0, n)
    df = None
    
    if rates is None or len(rates) < n:
        try:
            with sqlite3.connect(DB_NAME) as conn:
                cached_df = pd.read_sql(f"SELECT * FROM rates_M15 ORDER BY time DESC LIMIT {n}", conn)
                if not cached_df.empty:
                    cached_df = cached_df.sort_values('time')
                    if rates is not None:
                        df = pd.concat([cached_df, pd.DataFrame(rates)]).drop_duplicates('time').tail(n)
                    else:
                        df = cached_df
        except: pass
            
    if df is None and rates is not None:
        df = pd.DataFrame(rates)

    if df is None or len(df) < 200:
        return None
    
    # Update Cache
    try:
        with sqlite3.connect(DB_NAME) as conn:
            df.to_sql("rates_M15", conn, if_exists='replace', index=False)
    except: pass

    # Feature Engineering
    df['body'] = df['close'] - df['open']
    df['total_range'] = df['high'] - df['low'] + 1e-9
    df['body_ratio'] = df['body'] / df['total_range']
    df['upper_shadow_ratio'] = (df['high'] - df[['open','close']].max(axis=1)) / df['total_range']
    df['lower_shadow_ratio'] = (df[['open','close']].min(axis=1) - df['low']) / df['total_range']
    
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi() / 100.0
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    
    df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
    
    df['dist_ema20'] = (df['close'] - df['ema_20']) / (df['atr'] + 1e-9)
    df['dist_ema50'] = (df['close'] - df['ema_50']) / (df['atr'] + 1e-9)
    df['dist_ema200'] = (df['close'] - df['ema_200']) / (df['atr'] + 1e-9)
    
    bb = ta.volatility.BollingerBands(df['close'], window=20)
    df['bb_width'] = bb.bollinger_wband()
    df['bb_hband_dist'] = (bb.bollinger_hband() - df['close']) / (df['atr'] + 1e-9)
    df['bb_lband_dist'] = (df['close'] - bb.bollinger_lband()) / (df['atr'] + 1e-9)
    
    df['time_dt'] = pd.to_datetime(df['time'], unit='s')
    df['hour_sin'] = np.sin(2 * np.pi * df['time_dt'].dt.hour / 23.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['time_dt'].dt.hour / 23.0)
    df['day_sin'] = np.sin(2 * np.pi * df['time_dt'].dt.dayofweek / 6.0)
    df['day_cos'] = np.cos(2 * np.pi * df['time_dt'].dt.dayofweek / 6.0)
    
    return df.dropna().reset_index(drop=True)

def execute_order(action, confidence):
    """Sends a trade request to the MT5 terminal."""
    tick = mt5.symbol_info_tick(SYMBOL)
    if not tick: return
    
    price = tick.ask if action == "BUY" else tick.bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT_SIZE,
        "type": mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "magic": MAGIC_NUMBER,
        "comment": f"AI Pro {confidence:.1%}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Trade Failed: {result.comment}")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {action} Order Executed on {TRADING_MODE} at {price}")

def bot_main():
    """Main execution loop for the XAUUSD Pro AI Scalper."""
    print("="*50)
    print(f" XAUUSD ULTIMATE PRO AI SCALPER ({TRADING_MODE} MODE) ")
    print("="*50)
    
    if not init_mt5(): return
        
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading AI Brain...")
    scaler = joblib.load(SCALER_PATH)
    model = CandlePatternAI(input_size=len(FEATURE_COLS), seq_len=SEQ_LEN)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Strategy Active (Target {CONFIDENCE_THRESHOLD:.0%}+)")
    
    try:
        while True:
            # 1. Position Management (Quick Profit)
            positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)
            if positions:
                for pos in positions:
                    if pos.profit >= PROFIT_THRESHOLD:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Quick Profit Hit: ${pos.profit:.2f}. Closing #{pos.ticket}")
                        mt5.Close(SYMBOL, ticket=pos.ticket)

            # 2. AI Sequence Analysis
            df = get_latest_data()
            if df is not None and len(df) >= SEQ_LEN:
                # Pass directly without extra '.values' to keep feature names hint for sklearn if possible, 
                # but transform still returns array.
                recent_df = df[FEATURE_COLS].tail(SEQ_LEN)
                scaled_data = scaler.transform(recent_df)
                
                input_tensor = torch.FloatTensor(scaled_data).unsqueeze(0)
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    confidence, pred = torch.max(probs, 1)
                    
                action = {0: "BUY", 1: "SELL", 2: "HOLD"}[pred.item()]
                confidence = confidence.item()
                tick = mt5.symbol_info_tick(SYMBOL)
                
                # Feedback Log
                print(f"[{datetime.now().strftime('%H:%M:%S')}] AI Status: {action} ({confidence:.2%}) | Bid: {tick.bid if tick else 'N/A'}")
                
                # Execution Logic (Strictly respect Max Positions)
                if action in ["BUY", "SELL"] and confidence >= CONFIDENCE_THRESHOLD:
                    current_positions = len(positions) if positions else 0
                    if current_positions < MAX_OPEN_POSITIONS:
                        execute_order(action, confidence)
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Syncing market data...")
            
            time.sleep(15)
            
    except KeyboardInterrupt:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Bot stopped by operator.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    bot_main()
