import plotly.express as px
import plotly.graph_objects as go


def add_candle(fig, data, row = 1 , col =1):
    fig.add_trace(
        go.Candlestick(
            x = data.index ,
            open = data.open, close = data.close,
            high = data.high, low = data.low,
            name = data.symbol[0]
        ),
        row = row , col = col
    )

#    return fig


def add_line(fig, x, y, color, name, row = 1 , col =1):
    fig.add_trace(
        go.Scatter(
            x = x, y = y,
            name = name,
            line = dict(color = color, width = 1)
        ),
        row = row , col = col
    )


def add_mark(fig, x, y, color, name, row = 1 , col =1):
    fig.add_trace(
        go.Scatter(
            x = x, y = y,
            name = name,
            line = dict(color = color, width = 1),
            mode = "markers"
        ),
        row = row , col = col
    )
    
    