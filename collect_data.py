import MetaTrader5 as mt5
import pandas as pd
import sqlite3
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
SYMBOL = os.getenv("SYMBOL", "XAUUSD.vx")
DB_NAME = "trading_data.db"
TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M15": mt5.TIMEFRAME_M15
}

# Credentials
LOGIN = int(os.getenv("MT5_DEMO_LOGIN", 0))
PASSWORD = os.getenv("MT5_DEMO_PASSWORD", "")
SERVER = os.getenv("MT5_DEMO_SERVER", "")

def init_mt5():
    if not mt5.initialize():
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return False
    if not mt5.login(LOGIN, password=PASSWORD, server=SERVER):
        print(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False
    return True

def create_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    for tf_name in TIMEFRAMES.keys():
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS rates_{tf_name} (
                time INTEGER PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                tick_volume INTEGER,
                spread INTEGER,
                real_volume INTEGER
            )
        """)
    conn.commit()
    conn.close()

def collect_data(timeframe_name):
    tf = TIMEFRAMES[timeframe_name]
    
    # Chunked fetching to handle broker limits
    chunk_size = 180
    all_rates = []
    
    current_to = datetime.now()
    total_days = 730
    days_collected = 0
    
    while days_collected < total_days:
        current_from = current_to - timedelta(days=chunk_size)
        print(f"Fetching {chunk_size} days for {timeframe_name} (from {current_from} to {current_to})...")
        
        rates = mt5.copy_rates_range(SYMBOL, tf, current_from, current_to)
        
        if rates is not None and len(rates) > 0:
            all_rates.append(pd.DataFrame(rates))
            print(f"Successfully fetched {len(rates)} candles.")
            # Move window back
            current_to = current_from
            days_collected += chunk_size
        else:
            print(f"No more data or limit reached for {timeframe_name}. Error: {mt5.last_error()}")
            break

    if not all_rates:
        print(f"All collection attempts failed for {timeframe_name}")
        return
    
    # Concatenate all DataFrames
    df = pd.concat(all_rates)
    # Sort and remove duplicates
    df = df.drop_duplicates(subset=['time']).sort_values('time')
    
    # Connect to SQLite and save
    conn = sqlite3.connect(DB_NAME)
    df.to_sql(f"rates_{timeframe_name}", conn, if_exists='replace', index=False)
    
    # Ensure 'time' is index for better performance
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_time_{timeframe_name} ON rates_{timeframe_name} (time)")
    
    conn.commit()
    conn.close()
    print(f"Successfully saved {timeframe_name} data to {DB_NAME}")

def main():
    if not init_mt5():
        return
    
    create_db()
    
    # Check if symbol exists and is selected
    if not mt5.symbol_select(SYMBOL, True):
        print(f"Symbol {SYMBOL} not found or cannot be selected.")
        mt5.shutdown()
        return

    collect_data("M15")
    collect_data("M1")
    
    mt5.shutdown()

if __name__ == "__main__":
    main()
