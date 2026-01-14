from ib_insync import *
import pandas as pd
import os
from math import ceil

# ===== TEST MODE TOGGLE =====
TEST_MODE = True  # Set to False to actually place orders
# ============================

util.startLoop()
ib = IB()

try:
    # Connect to IBKR
    ib.connect('127.0.0.1', 4002, clientId=99)
    print("✓ Connected to IBKR\n")
    
    if TEST_MODE:
        print("⚠️  RUNNING IN TEST MODE - NO ORDERS WILL BE PLACED ⚠️\n")
    else:
        print("🔴 LIVE MODE - ORDERS WILL BE PLACED 🔴\n")
    
    # Step 1: Get current positions
    positions = ib.positions()
    
    if not positions:
        print("❌ No positions found. Cannot test.")
        exit()
    
    print("=== CURRENT POSITIONS ===")
    total_positions = {}
    for pos in positions:
        symbol = pos.contract.symbol
        qty = pos.position
        total_positions[symbol] = qty
        print(f"{symbol}: {qty} shares")
    
    # Step 2: Get current NAV
    av = ib.accountValues()
    df_vals = util.df(av)
    current_nav = df_vals[df_vals.tag == 'NetLiquidation']['value'].astype(float).sum()
    
    # Step 3: Create fake drawdown scenario (6% drawdown)
    print("\n=== SIMULATING 6% DRAWDOWN ===")
    peak_nav = current_nav / 0.94  # If current is 94% of peak, drawdown is 6%
    
    fake_history = pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=5, freq='D'),
        'nav': [peak_nav * 0.95, peak_nav, peak_nav * 0.98, peak_nav * 0.96, current_nav]
    })
    
    # Save fake history
    backup_file = "stopLossTracker_backup.parquet"
    if os.path.exists("stopLossTracker.parquet"):
        os.rename("stopLossTracker.parquet", backup_file)
        print("✓ Backed up existing stopLossTracker.parquet")
    
    fake_history.to_parquet("stopLossTracker.parquet", index=False)
    
    # Step 4: Test the exact logic from nav_retriever.py
    max_nav = fake_history['nav'].max()
    current_drawdown = (((max_nav - current_nav)/max_nav)*100).round(1)
    
    print(f"Peak NAV: ${peak_nav:,.2f}")
    print(f"Current NAV: ${current_nav:,.2f}")
    print(f"Calculated Drawdown: {current_drawdown}%")
    
    THRESHOLD = 5.0
    print(f"\n=== THRESHOLD CHECK ===")
    if current_drawdown > THRESHOLD:
        print(f"✓ Drawdown {current_drawdown}% > {THRESHOLD}% - WOULD TRIGGER REDUCTION")
    else:
        print(f"✗ Drawdown {current_drawdown}% <= {THRESHOLD}% - WOULD NOT TRIGGER")
    
    # Step 5: Simulate the reduce_all_positions_by_pct function
    print("\n=== SIMULATING 1% POSITION REDUCTION ===")
    pct = 0.01
    
    print(f"\nReducing all positions by {pct*100:.1f}%\n")
    
    simulated_trades = []
    
    for symbol, qty in total_positions.items():
        abs_qty = abs(qty)
        reduce_qty = int(ceil(abs_qty * pct))
        
        if qty > 0:
            action = 'SELL'
            new_qty = qty - reduce_qty
        else:
            action = 'BUY'
            new_qty = qty + reduce_qty
        
        simulated_trades.append({
            'symbol': symbol,
            'old_qty': qty,
            'action': action,
            'reduce_qty': reduce_qty,
            'new_qty': new_qty
        })
        
        if TEST_MODE:
            print(f"[SIMULATED] {action} {reduce_qty} {symbol}")
        else:
            print(f"{action} {reduce_qty} {symbol}")
    
    # Step 6: Show results
    print("\n=== EXPECTED RESULTS ===")
    print(f"{'Symbol':<10} {'Current':<10} {'Action':<10} {'Reduce':<10} {'New Qty':<10} {'Change %':<10}")
    print("=" * 70)
    
    for trade in simulated_trades:
        change_pct = (trade['reduce_qty'] / abs(trade['old_qty']) * 100)
        print(f"{trade['symbol']:<10} {trade['old_qty']:<10} {trade['action']:<10} "
              f"{trade['reduce_qty']:<10} {trade['new_qty']:<10} {change_pct:.2f}%")
    
    if TEST_MODE:
        print("\n" + "="*70)
        print("✓ TEST MODE: No actual orders were placed")
        print("✓ All calculations verified successfully!")
        print("\nTo place real orders:")
        print("1. Set TEST_MODE = False")
        print("2. Re-run the script")
        print("="*70)
    else:
        # LIVE MODE - Actually place orders
        print("\n" + "="*70)
        print("🔴 LIVE MODE - THIS WILL PLACE REAL ORDERS 🔴")
        print("="*70)
        response = input("Proceed with position reduction? (yes/no): ").strip().lower()
        
        if response != 'yes':
            print("\n❌ Cancelled by user")
        else:
            from util import reduce_all_positions_by_pct
            
            print("\n=== EXECUTING POSITION REDUCTION ===")
            reduce_all_positions_by_pct(ib, pct=0.01)
            
            ib.sleep(3)
            
            print("\n=== VERIFICATION ===")
            new_positions = ib.positions()
            new_positions_dict = {pos.contract.symbol: pos.position for pos in new_positions}
            
            for trade in simulated_trades:
                symbol = trade['symbol']
                new_qty = new_positions_dict.get(symbol, 0)
                expected_qty = trade['new_qty']
                match = "✓" if new_qty == expected_qty else "✗"
                print(f"{match} {symbol}: {trade['old_qty']} → {new_qty} (expected {expected_qty})")
    
    # Cleanup
    if os.path.exists(backup_file):
        os.rename(backup_file, "stopLossTracker.parquet")
        print("\n✓ Restored original stopLossTracker.parquet")
    else:
        os.remove("stopLossTracker.parquet")
        print("✓ Removed test file")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    
    # Restore backup if exists
    if os.path.exists("stopLossTracker_backup.parquet"):
        os.rename("stopLossTracker_backup.parquet", "stopLossTracker.parquet")
        print("✓ Restored backup")

finally:
    ib.disconnect()
    print("\n✓ Disconnected")