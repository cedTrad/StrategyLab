import pandas as pd
import numpy as np

# Calcul de la volatilité
def compute_daily_volatility(price_series, span=100):
    returns = price_series.pct_change()
    volatility = returns.ewm(span=span).std()
    return volatility


# Get events
def get_events(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False):
    #trgt = trgt[trgt > minRet]
    trgt = trgt.where(trgt >= minRet, minRet)
    if t1 is False: t1 = pd.Series(pd.NaT, index=tEvents)
    out = pd.DataFrame(index=tEvents)
    out['t1'] = t1.loc[out.index]
    out['trgt'] = trgt.loc[out.index]
    if ptSl[0] > 0: out['pt'] = ptSl[0] * out['trgt']
    else: out['pt'] = pd.Series(index=tEvents)
    if ptSl[1] > 0: out['sl'] = -ptSl[1] * out['trgt']
    else: out['sl'] = pd.Series(index=tEvents)
    return out


def apply_pt_sl_on_t1(close, events, ptSl, molecule):
    out = events.loc[molecule].copy(deep=True)
    if ptSl[0] > 0: pt = ptSl[0] * events['trgt']
    else: pt = pd.Series(index=events.index)
    if ptSl[1] > 0: sl = -ptSl[1] * events['trgt']
    else: sl = pd.Series(index=events.index)
    
    for loc, t1 in events['t1'].fillna(close.index[-1]).items():
        df0 = close[loc:t1]  # chemin des prix
        df0 = (df0 / close[loc] - 1)
        out.at[loc, 'sl'] = df0[df0 < sl[loc]].index.min()  # stop-loss
        out.at[loc, 'pt'] = df0[df0 > pt[loc]].index.min()  # prise de profit
    return out


def get_bins(events, close):
    events['out'] = np.nan
    first_touch = events[['sl', 'pt', 't1']].dropna(how='all').copy()
    first_touch['pt'] = pd.to_datetime(first_touch['pt'], errors='coerce')
    first_touch['sl'] = pd.to_datetime(first_touch['sl'], errors='coerce')
    first_touch = first_touch.min(axis=1)
    for loc, t in first_touch.items():
        if pd.isnull(t): continue
        if t == events.loc[loc, 't1']: events.at[loc, 'out'] = 0
        elif t == events.loc[loc, 'pt']: events.at[loc, 'out'] = 1
        elif t == events.loc[loc, 'sl']: events.at[loc, 'out'] = -1
    events['ret'] = np.log(close.shift(1) / close)
    return events

