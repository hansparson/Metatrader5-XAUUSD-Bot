import MetaTrader5 as mt5
import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
SYMBOL = os.getenv("SYMBOL", "XAUUSD.vx")
MAGIC_NUMBER = int(os.getenv("MAGIC_NUMBER", 123456))

def check_history():
    if not mt5.initialize():
        print("MT5 initialization failed")
        return

    # Fetch history from the last 24 hours
    from_date = datetime.now() - timedelta(days=1)
    to_date = datetime.now()

    # Get deals for the magic number
    deals = mt5.history_deals_get(from_date, to_date, group=f"*{SYMBOL}*")
    
    if deals is None or len(deals) == 0:
        print(f"No transactions found for {SYMBOL} with Magic Number {MAGIC_NUMBER} in the last 24 hours.")
        mt5.shutdown()
        return

    df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
    
    # Filter by magic number
    df = df[df['magic'] == MAGIC_NUMBER]
    
    # Filter only for deal types that involve profit/loss (DEAL_ENTRY_OUT or DEAL_ENTRY_INOUT)
    # Entry types: 0=IN, 1=OUT, 2=INOUT
    finished_trades = df[df['entry'].isin([1, 2])]

    if finished_trades.empty:
        print("No completed trades found yet.")
        mt5.shutdown()
        return

    total_profit = finished_trades['profit'].sum()
    total_commission = finished_trades['commission'].sum()
    total_swap = finished_trades['swap'].sum()
    net_profit = total_profit + total_commission + total_swap
    
    wins = finished_trades[finished_trades['profit'] > 0]
    losses = finished_trades[finished_trades['profit'] <= 0]

    print("="*50)
    print(f"TRADING HISTORY SUMMARY - {SYMBOL}")
    print(f"Period: {from_date.strftime('%Y-%m-%d %H:%M')} to {to_date.strftime('%Y-%m-%d %H:%M')}")
    print("="*50)
    print(f"Total Completed Trades : {len(finished_trades)}")
    print(f"Winning Trades         : {len(wins)}")
    print(f"Losing Trades          : {len(losses)}")
    print(f"Win Rate               : {(len(wins)/len(finished_trades)*100):.2f}%")
    print("-" * 50)
    print(f"Gross Profit/Loss      : ${total_profit:.2f}")
    print(f"Total Commissions      : ${total_commission:.2f}")
    print(f"Total Swap             : ${total_swap:.2f}")
    print(f"NET PROFIT/LOSS        : ${net_profit:.2f}")
    print("="*50)
    
    # Optional: Print last 5 trades
    print("\nLast 5 Completed Deals:")
    recent = finished_trades.tail(5)
    for index, row in recent.iterrows():
        deal_time = datetime.fromtimestamp(row['time']).strftime('%Y-%m-%d %H:%M:%S')
        deal_type = "BUY" if row['type'] == 1 else "SELL" # 1 for DEAL_TYPE_SELL (Closing a BUY), 0 for DEAL_TYPE_BUY (Closing a SELL)
        # Note: Deal type 1 means closing a buy position, deal type 0 means closing a sell position.
        print(f"Time: {deal_time} | Profit: ${row['profit']:.2f}")

    mt5.shutdown()

if __name__ == "__main__":
    check_history()
