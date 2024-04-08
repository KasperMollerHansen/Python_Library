# Import all necessary packages
import yfinance as yf
import warnings
import matplotlib.pyplot as plt

################################

def asign_color(row):
    if row['Open'] < row['Close']:
        return 'g'
    else:
        return 'r'

def download(stock,start=None, end=None,period=None):
    data = yf.download(stock, start=start, end=end,period=period)
    data['Color'] = data.apply(asign_color, axis=1)
    data.name = stock
    return data

def stock_plot(data,type="Close",ax = plt, grid = "True",title = "False"):
    if type == "Close":
        ax.plot(data.index[:],data["Close"], color = "Black")
    elif type == "Candle":        
        ax.vlines(data.index[:],data["High"],data["Low"],color = data["Color"],linewidth = 0.5)
        ax.vlines(data.index[:],data["Open"],data["Close"],color = data["Color"],linewidth = 2)
    else:
        warnings.warn("Missing valid argument")
    ax.tick_params(axis='x', rotation=45)
    if grid == "True":
        ax.grid()
    if title == "True":
        ax.set_title(data.name)
