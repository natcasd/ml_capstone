import pandas as pd
import yfinance as yf
import numpy as np
import torch
import numpy as np

def onehot(index, length):
  if index > length:
    print("index greater than length")
    return []
  arr = [0] * length
  arr[index] = 1
  return arr

def map_quarter(qtr):
  if len(qtr) == 2:
    return int(qtr[1])
  elif len(qtr) == 7:
    return (int(qtr[0:2]) - 1)//3 + 1
  else:
    return qtr

def get_volume_volatility_data(symbol, date, df):
  three_months_prior = date - pd.DateOffset(months=3)
  valid_data = df[(df['symbol'] == symbol) & (df['date'] > three_months_prior) & (df['date'] < date)]
  valid_data['returns'] = valid_data['close'].pct_change()
  daily_volatility = valid_data['returns'].std()
  annualized_volatility = daily_volatility * np.sqrt(252)
  avg_volume = valid_data['volume'].mean()
  normalized_volume = np.log(avg_volume + 1)
  return annualized_volatility, normalized_volume

def create_training_examples(symbols, earnings, symbol_to_marketcap, symbol_to_industry, symbol_to_sector, stock_prices):
  list_to_csv = []
  print(symbols)
  for symbol in symbols:
    stock_examples = []
    earnings_symbol = earnings[earnings["symbol"] == symbol]
    step = 7
    maxi = 0
    for start in range(len(earnings_symbol) - step):
      end = start + step
      unit = earnings_symbol.iloc[start:end]
      volatility, volume = get_volume_volatility_data(symbol, unit['date'].iloc[-1], stock_prices)
      time_series = unit[['eps', 'eps_est', 'qtr']].to_numpy()
      pred = time_series[-1][0] - time_series[-1][1]
      time_series = time_series[:-1].tolist()
      time_series = [ex[:-1] + onehot(int(ex[-1]-1), 4) for ex in time_series]
      if abs(pred) > maxi:
        maxi = abs(pred)
      volatility = torch.tensor(volatility)
      volume = torch.tensor(volume)
      marketcap = torch.tensor(symbol_to_marketcap[symbol])
      print(volume)
      sector = torch.tensor(symbol_to_sector[symbol])
      industry = torch.tensor(symbol_to_industry[symbol])
      time_series_data = torch.tensor(time_series, dtype=torch.float32)
      file_name = symbol + str(start) + ".pt"
      data = {"time_series": time_series_data, "volatility": volatility, "volume": volume, "marketcap": marketcap, "sector": sector, "industry": industry}
      torch.save(data, "training_data2/" + file_name)
      stock_examples.append([file_name, pred])

    stock_examples = [[i[0], i[1]/maxi] for i in stock_examples]
    list_to_csv += stock_examples
  
  df = pd.DataFrame(list_to_csv)
  df.to_csv('annotations2.csv', index=False)
              

def generate_stock_metadata_dicts(symbols):
  symbol_to_industry = {}
  symbol_to_sector = {}
  symbol_to_marketcap = {}
  for symbol in symbols:
    ticker = yf.Ticker(symbol)
    info = ticker.info
    if info and 'industry' in info and 'sector' in info and 'marketCap' in info:
        industry = info.get('industry', 'No industry data available')
        sector = info.get('sector', 'No sector data available')
        marketCap = info.get('marketCap', 'No market cap data available')
        
        if industry != 'No industry data available' and sector != 'No sector data available' and marketCap != "No market cap data available":
            symbol_to_industry[symbol] = industry
            symbol_to_sector[symbol] = sector
            symbol_to_marketcap[symbol] = marketCap     
    else:
        print(f"No valid data found for {symbol}")

  max_marketcap = max(symbol_to_marketcap.values())
  symbol_to_marketcap = {symbol: cap / max_marketcap for symbol, cap in symbol_to_marketcap.items()}

  industries = set(symbol_to_industry.values())
  industry_to_onehot = {}
  for index, industry in enumerate(industries):
    industry_to_onehot[industry] = onehot(index, len(industries))

  sectors = set(symbol_to_sector.values())
  sector_to_onehot = {}
  for index, sector in enumerate(sectors):
    sector_to_onehot[sector] = onehot(index, len(sectors))
  
  symbol_to_industry_onehot = {symbol: industry_to_onehot[industry] for symbol, industry in symbol_to_industry.items()}
  symbol_to_sector_onehot = {symbol: sector_to_onehot[sector] for symbol, sector in symbol_to_sector.items()}
  return symbol_to_industry_onehot, symbol_to_sector_onehot, symbol_to_marketcap


def main():   
  earnings = pd.read_csv("data/stocks_latest/earnings_latest.csv")

  earnings = earnings.dropna()
  earnings["qtr"] = earnings["qtr"].apply(map_quarter)
  earnings["date"] = pd.to_datetime(earnings["date"])
  symbol_counts = earnings["symbol"].value_counts()
  symbols = symbol_counts[symbol_counts >= 28].index.tolist()

  stock_prices = pd.read_csv("data/stocks_latest/stock_prices_latest.csv")
  stock_prices = stock_prices.sort_values(by=["symbol", "date"])
  stock_prices['date'] = pd.to_datetime(stock_prices['date'])

  symbol_to_industry, symbol_to_sector, symbol_to_marketcap = generate_stock_metadata_dicts(symbols)
  sorted_market_caps = sorted(symbol_to_marketcap.items(), key=lambda item: item[1], reverse=True)
    
  top100 = [symbol for symbol, market_cap in sorted_market_caps[:100]]

  create_training_examples(top100, earnings, symbol_to_marketcap, symbol_to_industry, symbol_to_sector, stock_prices)

if __name__ == '__main__':
    main()

  

