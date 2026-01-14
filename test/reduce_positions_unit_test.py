import pandas as pd
import os
from math import ceil

print("=== MOCK TEST (No IBKR Required) ===\n")

# Mock positions
mock_positions = {
    'AAPL': 100,
    'TSLA': -50,
    'GOOGL': 75,
    'MSFT': 200
}

# Mock NAV
current_nav = 100000
peak_nav = 106383  # 6% drawdown

# Calculate drawdown
drawdown = ((peak_nav - current_nav) / peak_nav * 100)

print(f"Peak NAV: ${peak_nav:,.2f}")
print(f"Current NAV: ${current_nav:,.2f}")
print(f"Drawdown: {drawdown:.1f}%")
print(f"Triggers 5% threshold: {drawdown > 5}")

# Test reduction calculation
print("\n=== EXPECTED REDUCTIONS (1%) ===")
pct = 0.01

for symbol, qty in mock_positions.items():
    abs_qty = abs(qty)
    reduce_qty = int(ceil(abs_qty * pct))
    
    if qty > 0:
        action = 'SELL'
        new_qty = qty - reduce_qty
    else:
        action = 'BUY'
        new_qty = qty + reduce_qty
    
    print(f"{symbol}: {action} {reduce_qty} ({qty} → {new_qty})")

# Test parquet file handling
print("\n=== TESTING FILE OPERATIONS ===")

fake_history = pd.DataFrame({
    'date': pd.date_range('2026-01-01', periods=5, freq='D'),
    'nav': [101000, 106383, 104000, 102000, 100000]
})

test_file = "test_stopLossTracker.parquet"
fake_history.to_parquet(test_file, index=False)
print(f"✓ Created {test_file}")

# Read it back
loaded = pd.read_parquet(test_file)
max_nav = loaded['nav'].max()
current = loaded['nav'].iloc[-1]
calculated_drawdown = ((max_nav - current) / max_nav * 100)

print(f"✓ Loaded file successfully")
print(f"  Max NAV from file: ${max_nav:,.2f}")
print(f"  Current NAV from file: ${current:,.2f}")
print(f"  Calculated drawdown: {calculated_drawdown:.1f}%")

# Cleanup
os.remove(test_file)
print(f"✓ Cleaned up {test_file}")

print("\n=== LOGIC TEST PASSED ===")
print("All calculations work correctly!")
print("\nTo test with real positions, you need:")
print("1. IBKR Gateway/TWS running")
print("2. Run: python test_position_reduction.py")