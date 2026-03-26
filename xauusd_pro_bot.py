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
MAX_OPEN_POSITIONS = 1 # Safety Hard-Lock
CONFIDENCE_THRESHOLD = float(os.getenv("PRO_CONFIDENCE_THRESHOLD", 0.45))
LOT_SIZE = float(os.getenv("PRO_LOT_SIZE", 0.05))
PROFIT_THRESHOLD = float(os.getenv("PRO_QUICK_PROFIT", 0.0))

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
    all_syms = [s.name for s in mt5.symbols_get()]
    for s in [SYMBOL_BASE, "XAUUSD.vx", "XAUUSD.vxc", "XAUUSD.m", "GOLD"]:
        if s in all_syms and mt5.symbol_select(s, True):
            SYMBOL = s; break
    
    info = mt5.symbol_info(SYMBOL)
    if info:
        if info.filling_mode & 1: FILLING_TYPE = mt5.ORDER_FILLING_FOK
        elif info.filling_mode & 2: FILLING_TYPE = mt5.ORDER_FILLING_IOC
        else: FILLING_TYPE = mt5.ORDER_FILLING_RETURN
    return True

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

def close_position(pos, reason="Reason"):
    tick = mt5.symbol_info_tick(SYMBOL)
    req = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": pos.volume,
        "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "position": pos.ticket, "price": tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask,
        "magic": MAGIC_NUMBER, "comment": f"Close: {reason}", "type_time": mt5.ORDER_TIME_GTC, "type_filling": FILLING_TYPE,
    }
    res = mt5.order_send(req)
    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ CLOSED #{pos.ticket} ({reason}) | PnL: {pos.profit:+.2f} USC")

def execute_order(action):
    tick = mt5.symbol_info_tick(SYMBOL)
    req = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": LOT_SIZE,
        "type": mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": tick.ask if action == "BUY" else tick.bid,
        "magic": MAGIC_NUMBER, "comment": "AI Entry", "type_time": mt5.ORDER_TIME_GTC, "type_filling": FILLING_TYPE,
    }
    res = mt5.order_send(req)
    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔥 {action} EXECUTED @ {res.price}")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ ERROR: {res.comment if res else 'Unknown'}")

def print_status_panel():
    account = mt5.account_info()
    if not account: return
    
    positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)
    floating_pnl = sum(p.profit for p in positions) if positions else 0.0
    
    print("-" * 65)
    print(f" 💰 Balance: {account.balance:.2f} USC | Equity: {account.equity:.2f} USC")
    print(f" 📈 Floating: {floating_pnl:+.2f} USC | Positions: {len(positions)}/{MAX_OPEN_POSITIONS}")
    print("-" * 65)

def bot_main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*65)
    print(f" 🚀 XAUUSD BRAIN V2.1 PRO - {TRADING_MODE} MODE ")
    print("="*65)
    
    if not init_mt5():
        print("❌ CRITICAL: MT5 Connection Failed.")
        return
        
    scaler, model = joblib.load(SCALER_PATH), CandlePatternAI(len(FEATURE_COLS), SEQ_LEN)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True)); model.eval()
    
    print(f"✅ AI Brain Loaded. Monitoring {SYMBOL}...")
    
    last_status_time = 0
    
    try:
        while True:
            # 1. Connection Health Check
            if not mt5.terminal_info().connected:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔴 CONNECTION LOST! Retrying...")
                mt5.shutdown(); time.sleep(5); init_mt5(); continue

            # 2. Periodic Status Update
            if time.time() - last_status_time > 60: # Every 1 minute
                print_status_panel()
                last_status_time = time.time()

            # 3. Market Analysis
            df = get_processed_data()
            if df is not None:
                recent = df[FEATURE_COLS].tail(SEQ_LEN)
                input_tensor = torch.FloatTensor(scaler.transform(recent)).unsqueeze(0)
                with torch.no_grad():
                    probs = torch.softmax(model(input_tensor), dim=1)
                    conf, pred = torch.max(probs, 1)
                
                action, conf = {0: "BUY", 1: "SELL", 2: "HOLD"}[pred.item()], conf.item()
                tick = mt5.symbol_info_tick(SYMBOL)
                
                # Dynamic Logging
                status_icon = "📉" if action == "SELL" else "📈" if action == "BUY" else "💤"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {status_icon} AI: {action:<4} ({conf:.1%}) | Price: {tick.bid}")

                # 4. Position Management
                positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)
                
                # Auto-Safety: Close excess positions
                if positions and len(positions) > MAX_OPEN_POSITIONS:
                    for p in positions[MAX_OPEN_POSITIONS:]: close_position(p, "Excess Safety")
                    positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)

                if positions:
                    for pos in positions:
                        if pos.profit <= -100.0:
                            close_position(pos, "Emergency SL")
                        elif (pos.type == 0 and action == "SELL") or (pos.type == 1 and action == "BUY"):
                            close_position(pos, "Signal Reversal")
                        elif PROFIT_THRESHOLD > 0 and pos.profit >= PROFIT_THRESHOLD:
                            close_position(pos, "Quick Profit")

                # 5. Entry Logic
                if action in ["BUY", "SELL"] and conf >= CONFIDENCE_THRESHOLD:
                    if not positions or len(positions) < MAX_OPEN_POSITIONS:
                        execute_order(action)
            
            time.sleep(15)
    except KeyboardInterrupt: pass
    finally: mt5.shutdown()

if __name__ == "__main__": bot_main()
