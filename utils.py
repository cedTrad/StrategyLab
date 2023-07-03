import numpy as np

import cvxpy
import scipy
import cvxopt

lambda_list = [0, 0.1, 0.5, 1, 2, 5, 10, 50, 200, 500, 1000, 2000, 5000, 10000, 100000]
lambda_value = [2000, 5000, 10000] # 10 000
#y = data['close'].to_numpy()

# trend filter : l1 et l2
def trend_filter(y, lambda_value, reg_norm):
    """ 
    reg_norm = 1 : L1 filter
    reg_norm = 2 : H-P filter
    """
    n = y.size
    
    ones_row = np.ones((1, n))

    # Creating matrix D
    D = scipy.sparse.spdiags(np.vstack((ones_row, -2*ones_row, ones_row)), range(3), n-2, n)
    
    # Solve
    solver = cvxpy.CVXOPT
    #solver = cvxpy.ECOS
    x = cvxpy.Variable(shape=n) 
    # x is the filtered trend that we initialize
    objective = cvxpy.Minimize(0.5 * cvxpy.sum_squares(y-x) 
                  + lambda_value * cvxpy.norm(D@x, reg_norm))
    problem = cvxpy.Problem(objective)
    
    try:
        problem.solve(solver=solver)
    except cvxpy.SolverError:
        lambda_value -= 100
        trend_filter(y, lambda_value, reg_norm)
    #print('lambda value : ',lambda_value)
    
    return x.value



class SMAVectorBacktester(object):
    
    def __init__(self, data, SMA1 , SMA2 , start, end):
        self.data = data
        self.SMA1 = SMA1
        self.SMA2 = SMA2
        self.start = start
        self.end = end
        self.result = None
        self.prepare_data(data)
    
    def prepare_data(self , data):
        self.data['return'] = np.log(self.data['close']/self.data['close'].shift(1))
        self.data['SMA1'] = self.data['close'].rolling(self.SMA1).mean()
        self.data['SMA2'] = self.data['close'].rolling(self.SMA2).mean()
    
    def set_parameters(self, SMA1=None , SMA2=None):
        ''' Updates SMA parameters and resp times series '''
        if SMA1 is not None:
            self.SMA1 = SMA1
            self.data['SMA1'] = self.data['close'].rolling(self.SMA1).mean()
        if SMA2 is not None:
            self.SMA2 = SMA2
            self.data['SMA2'] = self.data['close'].rolling(self.SMA2).mean()
        
    def run_strategy(self):
        ''' Backtests the training strategy'''
        data = self.data.copy().dropna()
        
        ''' Long position : 1 / Short position : -1'''
        data['position'] = np.where(data['SMA1']>data['SMA2'], 1, -1)
        data['strategy'] = data['position'].shift(1)*data['return']
        
        data.dropna(inplace = True)
        
        data['creturns'] = data['return'].cumsum().apply(np.exp)
        data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)
        self.results = data 
        
        # gross performance of the strategy
        aperf = data['cstrategy'].iloc[-1]
        
        # Out-/underperformance of strategy
        operf = aperf - data['creturns'].iloc[-1]
        return round(aperf,2) , round(operf , 2)
    
    def plot_results(self):
        if self.results is None:
            print("No results to plot yet . Run a strategy")
        title = " | SMA1=%d , SMA2=%d"%(self.SMA1 , self.SMA2)
        px.line(self.results[['creturns' , 'cstrategu']] , title = title)
        
    
    def update_and_run(self , SMA):
        ''' SMA : tuple '''
        self.set_parameters(int(SMA[0]) , int(SMA[1]))
        return self.run_strategy()[0]
    
    
    def optimize_parameters(self , SMA1_range, SMA2_range):
        ''' SMA1_range , SMA2_range : tuple ( start, end , step | size) '''
        opt = brute(self.update_and_run , (SMA1_range, SMA2_range), finish=None)
        return opt , self.update_and_run(opt)
    
    