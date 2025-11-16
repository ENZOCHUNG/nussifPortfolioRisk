from ib_insync import IB

ib = IB()

print("Connecting...")
try:
    ib.connect('127.0.0.1', 4002, clientId=999)
    print("✓ Connected to IB Gateway!")

    # Ping the API by requesting server time
    server_time = ib.reqCurrentTime()
    print("Server Time:", server_time)

except Exception as e:
    print("❌ Connection failed:", e)

finally:
    ib.disconnect()
    print("Disconnected.")
