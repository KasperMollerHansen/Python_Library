# Import all necessary packages
import yfinance as yf

################################

def asign_color(row):
    if row['Open'] < row['Close']:
        return 'g'
    else:
        return 'r'

def download(stock,period):
    data = yf.download("BAVA.CO", period="3d")
    data['Color'] = data.apply(asign_color, axis=1)
    return data

