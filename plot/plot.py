import plotly.express as px
import plotly.graph_objects as go


def add_candle(data):
    fig = go.Candlestick(
        x = data.index ,
        open = data.open, close = data.close,
        high = data.high, low = data.low,
        name = data.symbol[0]
    )
    return fig


def add_line(x, y, color, name):
    fig = go.Scatter(
        x = x, y = y,
        name = name,
        line = dict(color = color, width = 1)
    )
    return fig


def add_mark(x, y, color, name):
    fig = go.Scatter(
        x = x, y = y,
        name = name,
        line = dict(color = color, width = 1),
        mode = "markers"
    )
    return fig
    
    