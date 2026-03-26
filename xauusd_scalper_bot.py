import os
import time
import json
import requests
import pandas as pd
import MetaTrader5 as mt5
import ta
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
# Account placeholders using .env for seamless integration
ACCOUNT_MODE = os.getenv("ACCOUNT_MODE", "DEMO")
if ACCOUNT_MODE == "LIVE":
    LOGIN = int(os.getenv("MT5_LIVE_LOGIN", 0))
    PASSWORD = os.getenv("MT5_LIVE_PASSWORD", "")
    SERVER = os.getenv("MT5_LIVE_SERVER", "")
else:
    LOGIN = int(os.getenv("MT5_DEMO_LOGIN", 0))
    PASSWORD = os.getenv("MT5_DEMO_PASSWORD", "")
    SERVER = os.getenv("MT5_DEMO_SERVER", "")

SYMBOL = os.getenv("SYMBOL", "XAUUSD.vx") # from .env
MAGIC_NUMBER = int(os.getenv("MAGIC_NUMBER", 123456))
LOT_SIZE = float(os.getenv("DEFAULT_LOT", 0.01))

# Trading and filtering parameters
TIMEFRAME = mt5.TIMEFRAME_M1
MAX_CANDLES = 100
MAX_SPREAD_POINTS = 50 # 5 pips
CONFIDENCE_THRESHOLD = 50
MIN_QUICK_PROFIT = float(os.getenv("MIN_QUICK_PROFIT", 2.0)) # Profit in USD to trigger early close
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", 5)) # Allow multiple concurrent trades

# Ollama API Configuration
OLLAMA_URL = os.getenv("AI_URL", "http://127.0.0.1:11434/api/generate")
MODEL = "mistral" 


def init_mt5():
    """Initialize MetaTrader5 and connect to account."""
    if not mt5.initialize():
        print(f"[{datetime.now()}] MT5 initialization failed: {mt5.last_error()}")
        return False
        
    if LOGIN and PASSWORD and SERVER:
        if not mt5.login(LOGIN, password=PASSWORD, server=SERVER):
            print(f"[{datetime.now()}] MT5 login failed: {mt5.last_error()}")
            return False
            
    print(f"[{datetime.now()}] MT5 interconnected. Logged in to {LOGIN}.")

    # Configure symbol in Market Watch
    if not mt5.symbol_select(SYMBOL, True):
        print(f"[{datetime.now()}] Symbol {SYMBOL} not found.")
        return False
        
    return True


def get_data():
    """Fetch latest 100 candles for XAUUSD on M1."""
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, MAX_CANDLES)
    if rates is None or len(rates) == 0:
        return None
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def calculate_indicators(df):
    """Calculate EMA 9, EMA 21, and RSI (14) using pandas & ta."""
    if df is None or len(df) < 25:
        return None
        
    # EMA 9 and EMA 21
    df['ema_9'] = ta.trend.EMAIndicator(close=df['close'], window=9).ema_indicator()
    df['ema_21'] = ta.trend.EMAIndicator(close=df['close'], window=21).ema_indicator()
    
    # RSI 14
    df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    
    # We use the current/latest unfinished candle's data (or iloc[-2] for strict completed)
    return df.iloc[-1]


def get_market_condition(latest_data):
    """Determine trend structure and RSI filters."""
    ema9 = latest_data['ema_9']
    ema21 = latest_data['ema_21']
    rsi = latest_data['rsi_14']
    close = latest_data['close']
    
    # Determine trend
    if ema9 > ema21:
        trend = "uptrend"
    elif ema9 < ema21:
        trend = "downtrend"
    else:
        trend = "neutral"
        
    return {
        "price": close,
        "ema9": ema9,
        "ema21": ema21,
        "rsi": rsi,
        "trend": trend,
        "sideways": 45 <= rsi <= 55
    }


def get_ai_signal(market_cond):
    """Build prompt and send data to Ollama HTTP API."""
    prompt = f"""
You are an expert XAUUSD scalping bot. You are analyzing the 1-minute chart.
Market Data:
- Current Price: {market_cond['price']:.3f}
- EMA 9: {market_cond['ema9']:.3f}
- EMA 21: {market_cond['ema21']:.3f}
- RSI (14): {market_cond['rsi']:.1f}
- Trend Analysis: {market_cond['trend'].upper()}

Rules:
1. AGGRESSIVE SCALPING: You must actively look for trades. Do not be overly cautious.
2. RELAXED TREND: You can counter-trend if RSI is favorable (e.g., BUY in downtrend if RSI < 40, SELL in uptrend if RSI > 60).
3. Provide exact numerical price for Stop Loss (5 to 10 pips from entry).
4. Provide exact numerical price for Take Profit (10 to 15 pips from entry).
5. Bias towards ACTION. Only return NO_TRADE if the market is completely flat and directionless.
6. Return ONLY JSON, without code block markdown, pure JSON exactly as below:

{{
  "action": "BUY" | "SELL" | "NO_TRADE",
  "entry": {market_cond['price']:.3f},
  "sl": 0.0,
  "tp": 0.0,
  "confidence": 0,
  "reason": "brief string"
}}
"""
    
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        result_text = response.json().get("response", "").strip()
        
        # In case the model still outputs markdown code blocks
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]

        return json.loads(result_text.strip())
        
    except requests.exceptions.RequestException as e:
        print(f"[{datetime.now()}] Connection error with Ollama API: {e}")
    except json.JSONDecodeError as e:
        print(f"[{datetime.now()}] JSON parsing error from AI response: {e}\nResponse text: {result_text}")
    except Exception as e:
        print(f"[{datetime.now()}] General AI fetching error: {e}")
        
    return None


def check_and_close_profitable_positions():
    """Check open positions and close them if profit >= MIN_QUICK_PROFIT."""
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return
        
    for pos in positions:
        if pos.magic == MAGIC_NUMBER and pos.profit >= MIN_QUICK_PROFIT:
            print(f"[{datetime.now()}] QUICK PROFIT TRIGGERED: Closing #{pos.ticket} with Profit: ${pos.profit:.2f}")
            close_position(pos)


def close_position(pos):
    """Closes an open position."""
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return
        
    if pos.type == mt5.ORDER_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
        
    symbol_info = mt5.symbol_info(SYMBOL)
    filling_type = mt5.ORDER_FILLING_IOC
    if symbol_info.filling_mode & 1:
        filling_type = mt5.ORDER_FILLING_FOK
    elif symbol_info.filling_mode & 2:
        filling_type = mt5.ORDER_FILLING_IOC
    else:
        filling_type = mt5.ORDER_FILLING_RETURN
        
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pos.symbol,
        "volume": pos.volume,
        "type": order_type,
        "position": pos.ticket,
        "price": price,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "Quick Profit Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_type,
    }
    
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        retcode = result.retcode if result else "API Exception"
        print(f"[{datetime.now()}] Failed to close #{pos.ticket}. Error Code: {retcode}")
    else:
        print(f"[{datetime.now()}] Successfully closed #{pos.ticket} for Quick Profit!")


def has_open_positions():
    """Verify if max position holding limits are hit."""
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        return True # Default to true for safety on API read fail
        
    count = sum(1 for pos in positions if pos.magic == MAGIC_NUMBER)
    return count >= MAX_OPEN_POSITIONS


def check_spread():
    """Spread safety filter."""
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        return False
    return symbol_info.spread <= MAX_SPREAD_POINTS


def execute_trade(decision):
    """Place the BUY/SELL position using MT5 Order Send based on AI parameters."""
    action = decision.get("action")
    
    if action not in ["BUY", "SELL"]:
        return
        
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        print(f"[{datetime.now()}] Failed to get symbol info prior to order execution.")
        return
        
    if action == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(SYMBOL).ask
    else:  # SELL
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(SYMBOL).bid
        
    # Set filling mode based on symbol properties
    filling_type = mt5.ORDER_FILLING_IOC
    if symbol_info.filling_mode & 1:
        filling_type = mt5.ORDER_FILLING_FOK
    elif symbol_info.filling_mode & 2:
        filling_type = mt5.ORDER_FILLING_IOC
    else:
        filling_type = mt5.ORDER_FILLING_RETURN
        
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT_SIZE,
        "type": order_type,
        "price": price,
        "sl": float(decision.get("sl", 0.0)),
        "tp": float(decision.get("tp", 0.0)),
        "deviation": 20, # Acceptable slippage in points
        "magic": MAGIC_NUMBER,
        "comment": "AI Scalp XAUUSD",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_type,
    }
    
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        retcode = result.retcode if result else "API Exception"
        print(f"[{datetime.now()}] Order failed. Error Code: {retcode}")
    else:
        print(f"[{datetime.now()}] Trade execution successful! Action: {action} | Ticket: {result.order}")


def log_market_and_ai(market_cond, decision):
    """Console printout formatter for cyclic outputs."""
    print(f"\n--- [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {SYMBOL} M1 LOOP ---")
    print(f"MARKET : Price: {market_cond['price']:.2f} | Trend: {market_cond['trend']} | RSI: {market_cond['rsi']:.1f}")
    if market_cond['sideways']:
        print("FILTER : Sideways Market (RSI 45-55). No AI eval.")
    elif decision:
        print(f"AI RESP: {decision.get('action')} | Confidence: {decision.get('confidence')}% | SL: {decision.get('sl')} | TP: {decision.get('tp')}")
        print(f"REASON : {decision.get('reason')}")
    else:
        print("AI RESP: None / Parse Error")


def main_loop():
    """Constantly cycling event loop for automated bot operation."""
    print(f"Starting XAUUSD AI Scalper Bot - Waiting for initialization...")
    
    if not init_mt5():
        print("Critical initialization failure. Exiting.")
        return
        
    try:
        while True:
            # 0. Check open trades for quick profits
            check_and_close_profitable_positions()
            
            df = get_data()
            if df is not None:
                latest_data = calculate_indicators(df)
                
                if latest_data is not None:
                    market_cond = get_market_condition(latest_data)
                    
                    # 1. Sideways Filter Check
                    if market_cond['sideways']:
                        log_market_and_ai(market_cond, None)
                    else:
                        # 2. Get AI Decision
                        decision = get_ai_signal(market_cond)
                        log_market_and_ai(market_cond, decision)
                        
                        if decision and decision.get("action") in ["BUY", "SELL"]:
                            # 3. Confidence Threshold Check
                            if decision.get("confidence", 0) >= CONFIDENCE_THRESHOLD:
                                # 4. Max position limit check
                                if not has_open_positions():
                                    # 5. Spread check
                                    if check_spread():
                                        execute_trade(decision)
                                    else:
                                        print(f"[{datetime.now()}] SKIPPED: Spread too high.")
                                else:
                                    print(f"[{datetime.now()}] SKIPPED: Max open positions reached ({MAX_OPEN_POSITIONS}).")
                                    
                            else:
                                print(f"[{datetime.now()}] SKIPPED: Confidence {decision.get('confidence')}% < {CONFIDENCE_THRESHOLD}%")
            
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\nBot stopped by user. Shutting down...")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main_loop()
