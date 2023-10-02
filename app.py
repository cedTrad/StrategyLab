import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

from plot import subplot, plot_candle, add_line, add_line2

from feature.indicator import *
from data.get_data import *
from data.data import connect_db



st.set_page_config(page_title="Technical Analysis",
                    page_icon=":female-doctor",
                    layout="wide")

interval = "1d"
start = "2023"
end = "2023"

db = connect_db("database", interval)

symbol = st.selectbox("... Select ", ["BTC", "SOL", "ETH"])

data = db.get_data(symbol, start, end)
fig = subplot(nb_rows=2, nb_cols=1, row_heights=[0.8, 0.2])
plot_candle(fig=fig, col=1, row=1, data=data, name='ohlc')


tab_1, tab_2, tab_3, tab_4 = st.tabs(["Trend", "momentum", "Volality", "Volume"])

def slider(label):
    return st.slider(label, min_value=1, max_value=100)

def number(label):
    return st.number_input(label, min_value=1, max_value=100, value=14)

def multiselect(label, option):
    return st.multiselect(label, option)

def one_params(label, funct, col  = None, row  = None):
    period = number(label)
    x = funct(data, period)
    add_line2(fig, x, label, col=col, row=row)
    
#multiselect("", ["ema", "ma"])

with tab_1:
    with st.expander("... click ..."):
        """
        cols = st.columns(3)
        with cols[0]:
            one_params("ma_1", ma)
        with cols[1]:
            one_params("ma_2", ma)
        with cols[2]:
            one_params("ma_3", ma)
            """
        cols = st.columns(3)
        with cols[0]:
            one_params("ema_1", ema)
        with cols[1]:
            one_params("ema_2", ema)
        with cols[2]:
            one_params("ema_3", ema)




with tab_2:
    one_params("rsi", rsi, col=1, row=2)
    










fig.update_xaxes(showspikes=True)
fig.update_yaxes(showspikes=True)
fig.update_layout(height = 700 , width =1500,
                          legend = dict(orientation="h",
                                        yanchor="bottom", y=1,
                                        xanchor="right", x=0.5),
                          margin = {'t':0, 'b':0, 'l':10, 'r':0}
                          )
st.plotly_chart(fig, True)

# streamlit run c:/Users/cc/Desktop/CedAlgo/StrategyLab/app.py