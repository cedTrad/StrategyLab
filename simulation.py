from backtest.app import App

from threading import Thread
from multiprocessing import Process

class MApp(App):
    
    def __init__(self, symbols, capital, interval = "1d", start="2023", end="2023", db_trades = "simulation_"):
        App.__init__(self, symbols = symbols, capital = capital,
                     interval = interval, start = start, end = end, db_trades = db_trades)
        
    
    def export_data(self):
        self.asset_data.to_csv(f"data//data_{self.symbols}.csv")
        self.portfolio_data.to_csv(f"data//portfolio_{self.symbols}.csv")
        
        

def simule(params):
    app = MApp(**params)
    app.run(report = False)
    app.export_data()
    


portfolios = [
        {"symbols": ["ETH", "BTC"], "capital": 10000, "start": "2023-01-01", "end": "2023-01-31", "interval": "1d", "db_trades": "sim1"},
        {"symbols": ["EGLD", "XMR"], "capital": 15000, "start": "2023-02-01", "end": "2023-02-28", "interval": "1d", "db_trades": "sim2"}
    ]

"""
if __name__ == "__main__":
    processes = []
    for params in portfolios:
        p = Process(target = simule, args=(params,))
        #p = Thread(target = simule, args=(params,))
        processes.append(p)
        p.start()
            
    for p in processes:
        p.join()

    print("All simulations are completed.")
    
"""