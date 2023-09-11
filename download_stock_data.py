"""
  Very simple example for downloading historical closing/open prices for a given stock from the market
"""


import pandas as pd
import yfinance as yf

RIC_NAME = '^IXIC'  ## Argumento que se puede pasar por linea de comandos.

if __name__ == "__main__":
  ric_name  = yf.Ticker(RIC_NAME) 
  histtorical_data = msft.history(period="1mo").reset_index()
