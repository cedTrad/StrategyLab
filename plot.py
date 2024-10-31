import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_figure():
    fig = go.Figure()
    return fig


def subplots(nb_rows , nb_cols, row_heights = None, column_widths = None, vertical_spacing=0.01, horizontal_spacing = 0.02):
    fig = make_subplots(rows = nb_rows, cols = nb_cols, shared_xaxes = True, shared_yaxes = False,
                        row_heights = row_heights, column_widths= column_widths,
                        vertical_spacing = vertical_spacing, horizontal_spacing = horizontal_spacing,
                         specs = [
                             [{"secondary_y": True}],
                             [{"secondary_y": True}],
                             [{"secondary_y": True}]
                           ]
                         )
    return fig

def subplot(nb_rows , nb_cols, row_heights = None, column_widths = None, vertical_spacing=0.01, horizontal_spacing = 0.02):
    fig = make_subplots(rows = nb_rows, cols = nb_cols, shared_xaxes = True, shared_yaxes = False,
                        row_heights = row_heights, column_widths= column_widths,
                        vertical_spacing = vertical_spacing, horizontal_spacing = horizontal_spacing)
    return fig


# Plot candle
def plot_candle(fig, col, row, data, symbol):
    fig.add_trace(
        go.Candlestick(
            x = data.index , open = data.open, close = data.close,
            high = data.high, low = data.low, name = symbol,
        ),
        col = col, row = row
    )
    y_range = [min(data["low"]) * 0.99, max(data["high"]) * 1.01]
    fig = fig.update_yaxes(range=y_range, row=row, col=col)
    fig = fig.update_xaxes(rangeslider_visible=False)


def add_line(fig, data, feature, name, color = None, col = None, row = None):
    fig.add_trace(
        go.Scatter(
            x = data.index,
            y = data[feature],
            marker_color = color,
            name = name
        ),
        col = col, row = row
    )



def add_scatter(fig, data, feature, name, color = None, col = None, row = None, size=2):
    fig.add_trace(
        go.Scatter(
            x = data.index,
            y = data[feature],
            mode = 'markers',
            marker= dict(
                color=color, size=size, symbol='circle'
            ),
            name = name
        ),
        col = col, row = row
    )


def add_bar(fig, data, feature, name, color = None, col = None, row = None):
    fig.add_trace(
        go.Bar(
            x = data.index, y = data[feature],
            name = name,
            marker_color = color
        ),
        col = col, row = row
    )
    

def add_hline(fig, y, color=None, col=None, row=None):
    fig.add_hline(y = y , col = col, row = row, line_color = color)
    
def add_vline(fig, x , color=None, col=None, row=None):
    fig.add_vline(x = x , col = col, row = row, line_color = color)


def add_area(fig, data, color, feature, name, col = None , row = None):
    fig.add_trace(
        go.Scatter(
            x = data.index , y = data[feature],
            fill = "tozeroy",
            marker_color = color,
            name = name
        ),
        col = col , row = row
    )
    
    
def add_hist(fig, data, feature, name, col = None , row = None):
    fig.add_trace(
        go.Histogram(
            x = data[feature], histnorm = "probability", name = name
        ),
        col = col , row = row
    )



def add_second_y(fig, col, row, data, name = 'position'):
    fig.add_trace(
        go.Scatter(
            x = data.index,
            y = data[name],
            yaxis="y2",
            name = name
        ),
        col = col, row = row,
        secondary_y=True
    )



def add_second_y_candle(fig, col, row, data, name = 'position'):
    fig.add_trace(
        go.Scatter(
            x = data.index,
            y = data[name],
            yaxis="y2",
            name = name
        ),
        col = col, row = row,
        secondary_y=True
    )
    range = [min(data[name])*0.98, max(data[name])*1.02]
    fig.update_yaxes(range=range, row=row, col=col, secondary_y=True)




# --------------------------------------------------

def plot_macd(fig, data, row, col):
    # Ajouter l'histogramme MACD
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['macd_hist'],
            name='macd_hist',
            marker_color='black'
        ), row=row, col=col
    )
    
    # Ajouter la ligne MACD
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['macd'],
            mode='lines',
            name='MACD',
            line=dict(color='blue')
        ), row=row, col=col
    )
    
    # Ajouter la ligne de signal
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['macd_signal'],
            mode='lines',
            name='macd_signal',
            line=dict(color='red')
        ), row=row, col=col
    )
