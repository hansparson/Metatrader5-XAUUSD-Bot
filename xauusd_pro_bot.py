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
import warnings
from dotenv import load_dotenv
from model_v2 import CandlePatternAI

# Silence warnings
warnings.filterwarnings("ignore")
load_dotenv(override=True)

# =================================================================
# CONFIGURATION
# =================================================================
TRADING_MODE = os.getenv("TRADING_MODE", "DEMO").upper()
SYMBOL_BASE = os.getenv("PRO_SYMBOL", "XAUUSD.vxc")
TIMEFRAME_M15 = mt5.TIMEFRAME_M15
SEQ_LEN = int(os.getenv("PRO_SEQ_LEN", 50))
MODEL_PATH = os.getenv("PRO_MODEL_PATH", "xauusd_model_v2.pth")
SCALER_PATH = os.getenv("PRO_SCALER_PATH", "scaler.gz")
MAGIC_NUMBER = int(os.getenv("PRO_MAGIC_NUMBER", 20240326))

# CRITICAL SAFETY: Hard-limit positions to 1 for $3 balance
# We ignore .env here to force safety
MAX_OPEN_POSITIONS = 1 

PROFIT_THRESHOLD = float(os.getenv("PRO_QUICK_PROFIT", 0.0))
CONFIDENCE_THRESHOLD = float(os.getenv("PRO_CONFIDENCE_THRESHOLD", 0.45))
LOT_SIZE = float(os.getenv("PRO_LOT_SIZE", 0.05))

SYMBOL = SYMBOL_BASE
FILLING_TYPE = mt5.ORDER_FILLING_IOC

FEATURE_COLS = [
    'body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio',
    'rsi', 'atr', 'bb_width', 'bb_hband_dist', 'bb_lband_dist',
    'dist_ema20', 'dist_ema50', 'dist_ema200',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
]

# =================================================================
# CORE FUNCTIONS
# =================================================================

def init_mt5():
    if not mt5.initialize(): return False
    prefix = "MT5_LIVE_" if TRADING_MODE == "LIVE" else "MT5_DEMO_"
    login, pw, srv = os.getenv(prefix+"LOGIN"), os.getenv(prefix+"PASSWORD"), os.getenv(prefix+"SERVER")
    if login and not mt5.login(int(login), pw, srv): return False
    
    global SYMBOL, FILLING_TYPE
    all_symbols = [s.name for s in mt5.symbols_get()]
    for sym in [SYMBOL_BASE, "XAUUSD.vx", "XAUUSD.vxc", "GOLD"]:
        if sym in all_symbols and mt5.symbol_select(sym, True):
            SYMBOL = sym; break
    
    info = mt5.symbol_info(SYMBOL)
    if info:
        if info.filling_mode & 1: FILLING_TYPE = mt5.ORDER_FILLING_FOK
        elif info.filling_mode & 2: FILLING_TYPE = mt5.ORDER_FILLING_IOC
        else: FILLING_TYPE = mt5.ORDER_FILLING_RETURN
    return True

def close_position(pos):
    tick = mt5.symbol_info_tick(SYMBOL)
    req = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": pos.volume,
        "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "position": pos.ticket, "price": tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask,
        "magic": MAGIC_NUMBER, "comment": "Safety Exit", "type_time": mt5.ORDER_TIME_GTC, "type_filling": FILLING_TYPE,
    }
    res = mt5.order_send(req)
    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Closed #{pos.ticket} | Profit: ${pos.profit:.2f}")

def get_processed_data():
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME_M15, 0, 300)
    if rates is None or len(rates) < 200: return None
    df = pd.DataFrame(rates)
    df['total_range'] = df['high'] - df['low'] + 1e-9
    df['body_ratio'] = (df['close'] - df['open']) / df['total_range']
    df['upper_shadow_ratio'] = (df['high'] - df[['open','close']].max(axis=1)) / df['total_range']
    df['lower_shadow_ratio'] = (df[['open','close']].min(axis=1) - df['low']) / df['total_range']
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi() / 100.0
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    ema20, ema50, ema200 = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator(), ta.trend.EMAIndicator(df['close'], window=50).ema_indicator(), ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
    df['dist_ema20'], df['dist_ema50'], df['dist_ema200'] = (df['close']-ema20)/(df['atr']+1e-9), (df['close']-ema50)/(df['atr']+1e-9), (df['close']-ema200)/(df['atr']+1e-9)
    bb = ta.volatility.BollingerBands(df['close'], window=20)
    df['bb_width'], df['bb_hband_dist'], df['bb_lband_dist'] = bb.bollinger_wband(), (bb.bollinger_hband()-df['close'])/(df['atr']+1e-9), (df['close']-bb.bollinger_lband())/(df['atr']+1e-9)
    df['time_dt'] = pd.to_datetime(df['time'], unit='s')
    df['hour_sin'], df['hour_cos'] = np.sin(2*np.pi*df['time_dt'].dt.hour/23.0), np.cos(2*np.pi*df['time_dt'].dt.hour/23.0)
    df['day_sin'], df['day_cos'] = np.sin(2*np.pi*df['time_dt'].dt.dayofweek/6.0), np.cos(2*np.pi*df['time_dt'].dt.dayofweek/6.0)
    return df.dropna().reset_index(drop=True)

def execute_order(action):
    tick = mt5.symbol_info_tick(SYMBOL)
    req = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": LOT_SIZE,
        "type": mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": tick.ask if action == "BUY" else tick.bid,
        "magic": MAGIC_NUMBER, "comment": "AI V2.1 Safety", "type_time": mt5.ORDER_TIME_GTC, "type_filling": FILLING_TYPE,
    }
    res = mt5.order_send(req)
    if res and res.retcode == mt5.TRADE_RETCODE_DONE: print(f"[{datetime.now().strftime('%H:%M:%S')}] {action} Executed (Lot {LOT_SIZE})")

def bot_main():
    print("="*50); print(f" XAUUSD SAFETY ENGINE V2.1 "); print("="*50)
    if not init_mt5(): return
    scaler, model = joblib.load(SCALER_PATH), CandlePatternAI(len(FEATURE_COLS), SEQ_LEN)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True)); model.eval()
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Mode: {TRADING_MODE} | Limit: {MAX_OPEN_POSITIONS} Pos")
    
    try:
        while True:
            df = get_processed_data()
            if df is not None:
                recent = df[FEATURE_COLS].tail(SEQ_LEN)
                input_tensor = torch.FloatTensor(scaler.transform(recent)).unsqueeze(0)
                with torch.no_grad():
                    probs = torch.softmax(model(input_tensor), dim=1)
                    conf, pred = torch.max(probs, 1)
                
                action, conf = {0: "BUY", 1: "SELL", 2: "HOLD"}[pred.item()], conf.item()
                tick = mt5.symbol_info_tick(SYMBOL)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] AI: {action} ({conf:.1%}) | {SYMBOL}: {tick.bid}")

                positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)
                
                # URGENT SAFETY: Close excess positions if somehow opened
                if positions and len(positions) > MAX_OPEN_POSITIONS:
                    print(f"[!] Over-exposure detected! Closing excess positions...")
                    for p in positions[MAX_OPEN_POSITIONS:]: close_position(p)
                    positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)

                if positions:
                    for pos in positions:
                        if pos.profit <= -100.0:
                            print(f"[!] Stop Loss Hit. Closing #{pos.ticket}")
                            close_position(pos)
                        elif (pos.type == 0 and action == "SELL") or (pos.type == 1 and action == "BUY"):
                            print(f"[EXIT] Signal reversed. Closing #{pos.ticket}")
                            close_position(pos)
                        elif PROFIT_THRESHOLD > 0 and pos.profit >= PROFIT_THRESHOLD:
                            close_position(pos)

                if action in ["BUY", "SELL"] and conf >= CONFIDENCE_THRESHOLD:
                    if not positions or len(positions) < MAX_OPEN_POSITIONS:
                        execute_order(action)
            time.sleep(15)
    except KeyboardInterrupt: pass
    finally: mt5.shutdown()

if __name__ == "__main__": bot_main()
