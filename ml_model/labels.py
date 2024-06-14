import pandas as pd
import numpy as np

def apply_triple_barrier(prices, events, profit_mult, loss_mult, time_barrier):
    """
    prices: Series des prix
    events: DataFrame avec les colonnes 't1' pour les horizons temporels et 'trgt' pour les seuils
    profit_mult, loss_mult: Multiplicateurs pour définir les seuils de profit et de perte
    time_barrier: Nombre de jours pour la barrière verticale
    """
    # Stocker les temps de toucher pour chaque barrière
    out = events[['t1']].copy(deep=True)
    out['touch_time'] = np.nan
    out['label'] = 0  # Par défaut, la barrière de temps est atteinte sans toucher les barrières de profit/perte
    
    for ix, event in events.iterrows():
        start_price = prices.loc[ix]
        end_date = event['t1']
        target = event['trgt']
        
        if pd.isna(start_price) or pd.isna(target):
            continue
        
        # Définir les seuils
        upper_barrier = start_price * (1 + profit_mult * target)
        lower_barrier = start_price * (1 - loss_mult * target)
        
        # Filtrer les prix dans la fenêtre de temps
        price_sub = prices.loc[ix:end_date]
        
        # Vérifier le toucher des barrières supérieure et inférieure
        upper_touch = price_sub[price_sub >= upper_barrier].index.min()
        lower_touch = price_sub[price_sub <= lower_barrier].index.min()
        
        # Déterminer le premier toucher
        first_touch = min(filter(pd.notna, [upper_touch, lower_touch, end_date]))
        if pd.notna(first_touch):
            out.at[ix, 'touch_time'] = first_touch
            if first_touch == upper_touch:
                out.at[ix, 'label'] = 1
            elif first_touch == lower_touch:
                out.at[ix, 'label'] = -1

    return out
