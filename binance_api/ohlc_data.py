from binance.client import Client




def get_ohlc_data(symbol='ETHUSDT', interval='1m'):
    client = Client()
    candles = client.get_klines(symbol=symbol, interval=interval, limit=1)
    ohlc_data = {
        'open': float(candles[0][1]),
        'high': float(candles[0][2]),
        'low': float(candles[0][3]),
        'close': float(candles[0][4]),
        'volume': float(candles[0][5])
    }
    return ohlc_data

