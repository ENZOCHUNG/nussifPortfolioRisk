from math import ceil
from ib_insync import *


def reduce_all_positions_by_pct(ib: IB, pct: float = 0.01):
    """
    Reduce every open position by pct (e.g. 0.01 = 1%)
    """
    positions = ib.positions()

    if not positions:
        print("No open positions.")
        return

    print(f"Reducing all positions by {pct*100:.1f}%")

    trades = []

    for pos in positions:
        contract = pos.contract
        qty = pos.position

        abs_qty = abs(qty)
        reduce_qty = int(ceil(abs_qty * pct))

        if qty > 0:
            action = 'SELL'
        else:
            action = 'BUY'

        order = MarketOrder(action, reduce_qty)

        trade = ib.placeOrder(contract, order)
        trades.append(trade)

        print(f"{action} {reduce_qty} {contract.symbol}")

    # Wait until all orders complete
    ib.waitOnUpdate()
    for t in trades:
        t.filledEvent += lambda trade: print(
            f"Filled {trade.order.action} {trade.order.totalQuantity} {trade.contract.symbol}"
        )

    print("Position reduction orders submitted.")
