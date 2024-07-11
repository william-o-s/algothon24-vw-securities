import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class ineffecieintStocks:
    def __init__(self, raw_data, ma_period=21, ema_period=9, no_sd=2):
        self.data = pd.DataFrame(raw_data.T)
        self.stocks_dict = {}
        # populate dictionary with individual stocks
        for col in self.data:
            self.stocks_dict[col] = self.data[col]
        self.ma_period = ma_period
        self.ema_period = ema_period
        self.no_sd = no_sd

    def raw(self):
        # Create directory if it doesn't exist
        output_dir = Path("Raw Data 750 Day")
        output_dir.mkdir(parents=True, exist_ok=True)

        for id, stock_price in self.stocks_dict.items():
            plt.figure(figsize=(12, 6))
            plt.plot(stock_price, label=f'Stock {id}', linewidth=0.5)
            plt.title(f'Stock {id} Price')
            plt.xlabel('Day')
            plt.ylabel('Price')
            plt.legend(loc='upper left', fontsize='small')
            plt.grid(True)

            # Save the plot to the directory
            plot_path = output_dir / f'stock{id}.png'
            plt.savefig(plot_path)
            plt.close()

    def bbCalc(self) -> dict:
        dictionary = {}
        for id, stock_price in self.stocks_dict.items():
            ma = stock_price.rolling(window=self.ma_period).mean().dropna()
            sd = stock_price.rolling(window=self.ma_period).std().dropna()
            upper_band = ma + (self.no_sd*sd)
            lower_band = ma - (self.no_sd*sd)

            bollinger_bands = pd.DataFrame({
            'Price': stock_price,
            'Moving Average': ma,
            'Upper Band': upper_band,
            'Lower Band': lower_band
            })

            dictionary[id] = bollinger_bands
        
        return dictionary

    def bbGraph(self):
        # Create directory if it doesn't exist
        output_dir = Path("Bollinger Bands")
        output_dir.mkdir(parents=True, exist_ok=True)
        bollinger_bands_dict = self.bbCalc()

        for id, bollinger_bands in bollinger_bands_dict.items():
            # Plotting the Bollinger Bands
            plt.figure(figsize=(12, 6))
            plt.plot(bollinger_bands['Price'], label='Price')
            plt.plot(bollinger_bands['Moving Average'], label='Moving Average', linestyle='--')
            plt.plot(bollinger_bands['Upper Band'], label='Upper Band', linestyle='--')
            plt.plot(bollinger_bands['Lower Band'], label='Lower Band', linestyle='--')
            plt.fill_between(bollinger_bands.index, bollinger_bands['Upper Band'], bollinger_bands['Lower Band'], color='gray', alpha=0.3)

            # Add a vertical line at id=250
            plt.axvline(x=250, color='red', linestyle='--', linewidth=1)

            plt.title(f'Stock {id} - {self.ma_period}MA Bollinger Bands')
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.legend()
            plt.grid()

            # Save the plot to the directory
            plot_path = output_dir / f'stock{id}_{self.ma_period}MA_BB.png'
            plt.savefig(plot_path)
            plt.close()

    def goldenCrossCalc(self) -> dict:
        dictionary = {}
        for id, stock_price in self.stocks_dict.items():
            ma50 = stock_price.rolling(window=50).mean().dropna()
            ma200 = stock_price.rolling(window=200).mean().dropna()

            goldenCross = pd.DataFrame({
                'Price': stock_price,
                '50 Day MA': ma50,
                '200 Day MA': ma200 
            })

            dictionary[id] = goldenCross
        
        return dictionary
    
    def goldenCrossGraph(self):
        # Create directory if it doesn't exist
        output_dir = Path("Golden Cross")
        output_dir.mkdir(parents=True, exist_ok=True)
        golden_cross_dict = self.goldenCrossCalc()

        for id, golden_cross in golden_cross_dict.items():
            # Plotting the Bollinger Bands
            plt.figure(figsize=(12, 6))
            plt.plot(golden_cross['Price'], label='Price')
            plt.plot(golden_cross['50 Day MA'], label='50 Day MA')
            plt.plot(golden_cross['200 Day MA'], label='200 Day MA')

            # Add a vertical line at id=250
            plt.axvline(x=250, color='red', linestyle='--', linewidth=1)

            plt.title(f'Stock {id} - Golden Crossover')
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.legend()
            plt.grid()

            # Save the plot to the directory
            plot_path = output_dir / f'stock{id}_GC.png'
            plt.savefig(plot_path)
            plt.close()

    def rsiCalc(self, window=14) -> dict:
        dictionary = {}
        for id, stock_price in self.stocks_dict.items():
            delta = stock_price.diff()

            # seperate gains and losses
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=window, min_periods=1).mean()
            avg_loss = loss.rolling(window=window, min_periods=1).mean()

            rs = avg_gain/avg_loss

            rsi = 100 - (100 / (1+rs))

            rsi_df = pd.DataFrame({
            'Price': stock_price,
            'RSI': rsi
            })

            dictionary[id] = rsi_df
        
        return dictionary
    
    def rsiGraph(self):
        # Create directory if it doesn't exist
        output_dir = Path("Relative Strength Index")
        output_dir.mkdir(parents=True, exist_ok=True)
        rsi_dict = self.rsiCalc()

        for id, rsi in rsi_dict.items():
            plt.figure(figsize=(12, 6))
            
            # Plotting the price
            plt.subplot(2,1,1)
            plt.plot(rsi['Price'], label='Price')
            # Add a vertical line at id=250
            plt.axvline(x=250, color='red', linestyle='--', linewidth=1)

            plt.title(f'Stock {id} - RSI')
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.legend()
            plt.grid()

            # plotting the rsi
            plt.subplot(2,1,2)
            plt.plot(rsi['RSI'], label='RSI', color='blue')
            plt.axhline(y=30, color='red', linestyle='--', label='Oversold (30)')
            plt.axhline(y=70, color='green', linestyle='--', label='Overbought (70)')
            plt.ylabel('RSI')
            plt.grid()
            plt.tight_layout()

            # Save the plot to the directory
            plot_path = output_dir / f'stock{id}_RSI.png'
            plt.savefig(plot_path)
            plt.close()

    def stochRSICalc(self, rsi_d, window=14) -> dict:
        dictionary = {}
        rsi_dict = rsi_d
        for id, stock in rsi_dict.items():
            rsi = stock['RSI']
            stoch_rsi = (rsi - rsi.rolling(window=window, min_periods=1).min()) / (rsi.rolling(window=window, min_periods=1).max() - rsi.rolling(window=window, min_periods=1).min())*100
            
            stoch_rsi_df = pd.DataFrame({
            'Price': stock['Price'],
            'Stochastic RSI': stoch_rsi
            })

            dictionary[id] = stoch_rsi_df
        
        return dictionary

    def stochRSIGraph(self):
        # Create directory if it doesn't exist
        output_dir = Path("Stochastic RSI")
        output_dir.mkdir(parents=True, exist_ok=True)
        stoch_rsi_dict = self.stochRSICalc(self.rsiCalc())

        for id, rsi in stoch_rsi_dict.items():
            plt.figure(figsize=(12, 6))
            
            # Plotting the price
            plt.subplot(2,1,1)
            plt.plot(rsi['Price'], label='Price')
            # Add a vertical line at id=250
            plt.axvline(x=250, color='red', linestyle='--', linewidth=1)

            plt.title(f'Stock {id} - Stochastic RSI')
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.legend()
            plt.grid()

            # plotting the rsi
            plt.subplot(2,1,2)
            plt.plot(rsi['Stochastic RSI'], label='Stochastic RSI', color='blue')
            plt.axhline(y=20, color='red', linestyle='--', label='Oversold (20)')
            plt.axhline(y=80, color='green', linestyle='--', label='Overbought (80)')
            plt.ylabel('RSI')
            plt.grid()
            plt.tight_layout()

            # Save the plot to the directory
            plot_path = output_dir / f'stock{id}_Stochastic_RSI.png'
            plt.savefig(plot_path)
            plt.close()

    def macdCalc(self) -> dict:
        dictionary = {}
        for id, stock_price in self.stocks_dict.items():
            short_ema = stock_price.ewm(span=12, adjust=False).mean()
            long_ema = stock_price.ewm(span=26, adjust=False).mean()
            macd_line = short_ema - long_ema

            # calculate signal line
            signal_line = macd_line.ewm(span=9, adjust=False).mean()

            # calculate histogram
            macd_histogram = macd_line - signal_line

            # separate positive and negative values for histogram
            pos_hist = macd_histogram[macd_histogram >= 0]
            neg_hist = macd_histogram[macd_histogram < 0]

            macd_df = pd.DataFrame({
                'Price': stock_price,
                'MACD Line': macd_line,
                'Signal Line': signal_line,
                'Positive Histogram': pos_hist,
                'Negative Histogram': neg_hist
            })

            dictionary[id] = macd_df

        return dictionary

    def macdGraph(self):
        # Create directory if it doesn't exist
        output_dir = Path("MACD")
        output_dir.mkdir(parents=True, exist_ok=True)
        macd_dict = self.macdCalc()

        for id, macd in macd_dict.items():
            plt.figure(figsize=(12, 6))
            
            # Plotting the price
            plt.subplot(2,1,1)
            plt.plot(macd['Price'], label='Price')
            # Add a vertical line at id=250
            plt.axvline(x=250, color='red', linestyle='--', linewidth=1)

            plt.title(f'Stock {id} - MACD')
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.legend()
            plt.grid()

            plt.subplot(2,1,2)
            plt.plot(macd['MACD Line'], label='MACD Line', color='blue')
            plt.plot(macd['Signal Line'], label='Signal Line', color='red')
            plt.bar(macd['Positive Histogram'].index, macd['Positive Histogram'], label='Positive Histogram', color='green', alpha=0.5)
            plt.bar(macd['Negative Histogram'].index, macd['Negative Histogram'], label='Negative Histogram', color='red', alpha=0.5)

            plt.title('MACD Indicator')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend(loc='upper left')
            plt.grid()
            plt.tight_layout()
            
            # Save the plot to the directory
            plot_path = output_dir / f'stock{id}_MACD.png'
            plt.savefig(plot_path)
            plt.close()
    
    def atrCalc(self, window=14):
        dictionary = {}
        for id, stock_price in self.stocks_dict.items():
            tr = stock_price.diff().abs()
            atr = tr.rolling(window=window).mean()

            atr_df = pd.DataFrame({
                'Price': stock_price,
                'ATR': atr
            })

            dictionary[id] = atr_df
        
        return dictionary

    def everythingGraph(self):
        # Create directory if it doesn't exist
        output_dir = Path("All Indicators")
        output_dir.mkdir(parents=True, exist_ok=True)
        bollinger_bands_dict = self.bbCalc()
        stoch_rsi_dict = self.stochRSICalc(self.rsiCalc())
        rsi_dict = self.rsiCalc()
        macd_dict = self.macdCalc()
        atr_dict = self.atrCalc(200)
        gc_dict = self.goldenCrossCalc()
        
        stock_ids = bollinger_bands_dict.keys()

        for stock_id in stock_ids:
            plt.figure(figsize=(14, 14))
            # graphing BB and Price
            if stock_id in bollinger_bands_dict:
                bollinger_bands = bollinger_bands_dict[stock_id]
                golden_cross = gc_dict[stock_id]
                # Plotting the Bollinger Bands
                plt.subplot(5,1,1)
                plt.plot(bollinger_bands['Price'], label='Price')
                plt.plot(bollinger_bands['Moving Average'], label='Moving Average', linestyle='--')
                plt.plot(bollinger_bands['Upper Band'], label='Upper Band', linestyle='--')
                plt.plot(bollinger_bands['Lower Band'], label='Lower Band', linestyle='--')
                plt.fill_between(bollinger_bands.index, bollinger_bands['Upper Band'], bollinger_bands['Lower Band'], color='gray', alpha=0.3)
                plt.plot(golden_cross['50 Day MA'], label='50 Day MA')
                plt.plot(golden_cross['200 Day MA'], label='200 Day MA')

                # Add a vertical line at id=250
                plt.axvline(x=250, color='red', linestyle='--', linewidth=1)

                plt.title(f'Stock {stock_id}')
                plt.xlabel('Days')
                plt.ylabel('Price ($)')
                plt.legend()
                plt.grid()
            
            # graphing Stochastic Relative Strength Index
            if stock_id in stoch_rsi_dict:
                srsi = stoch_rsi_dict[stock_id]
                plt.subplot(5,1,2)
                # plotting the rsi
                plt.plot(srsi['Stochastic RSI'], label='Stochastic RSI', color='blue')
                plt.axhline(y=20, color='red', linestyle='--', label='Oversold (20)')
                plt.axhline(y=80, color='green', linestyle='--', label='Overbought (80)')
                # Add a vertical line at id=250
                plt.axvline(x=250, color='red', linestyle='--', linewidth=1)
                plt.ylabel('Stoch RSI')
                plt.grid()

            # graphing Relative Strength Index
            if stock_id in rsi_dict:
                rsi = rsi_dict[stock_id]
                # plotting the rsi
                plt.subplot(5,1,3)
                plt.plot(rsi['RSI'], label='RSI', color='blue')
                plt.axhline(y=30, color='red', linestyle='--', label='Oversold (30)')
                plt.axhline(y=70, color='green', linestyle='--', label='Overbought (70)')
                # Add a vertical line at id=250
                plt.axvline(x=250, color='red', linestyle='--', linewidth=1)
                plt.ylabel('RSI')
                plt.grid()

            # grpahing macd
            if stock_id in macd_dict:
                macd = macd_dict[stock_id]
                plt.subplot(5,1,4)
                plt.plot(macd['MACD Line'], label='MACD Line', color='blue')
                plt.plot(macd['Signal Line'], label='Signal Line', color='red')
                plt.bar(macd['Positive Histogram'].index, macd['Positive Histogram'], label='Positive Histogram', color='green', alpha=0.5)
                plt.bar(macd['Negative Histogram'].index, macd['Negative Histogram'], label='Negative Histogram', color='red', alpha=0.5)

                # Add a vertical line at id=250
                plt.axvline(x=250, color='red', linestyle='--', linewidth=1)

                plt.title('MACD Indicator')
                plt.legend(loc='upper left')
                plt.grid()

            if stock_id in atr_dict:
                atr = atr_dict[stock_id]
                plt.subplot(5,1,5)
                plt.plot(atr['ATR'], label='ATR')

                plt.ylabel('ATR')
                plt.grid()

            # Save the plot to the directory
            plt.tight_layout()
            plot_path = output_dir / f'stock{stock_id}.png'
            plt.savefig(plot_path)
            plt.close()
            
class Stocks:
    def __init__(self, raw_data):
        self.data = pd.DataFrame(raw_data.T).stack().reset_index()
        self.data.columns = ['Day', 'Stock', 'Price']
        self.whatToGraph = []

    def raw(self):
        # Create directory if it doesn't exist
        output_dir = Path("Raw Data")
        output_dir.mkdir(parents=True, exist_ok=True)

        for id, stock_price in self.stocks_dict.items():
            plt.figure(figsize=(12, 6))
            plt.plot(stock_price, label=f'Stock {id}', linewidth=0.5)
            plt.title(f'Stock {id} Price')
            plt.xlabel('Day')
            plt.ylabel('Price')
            plt.legend(loc='upper left', fontsize='small')
            plt.grid(True)

            # Save the plot to the directory
            plot_path = output_dir / f'stock{id}.png'
            plt.savefig(plot_path)
            plt.close()

    def bbCalc(self, ma_period=21):
        # Calculate the moving average, standard deviation, and Bollinger Bands
        self.data[f'{ma_period}MA'] = self.data.groupby('Stock')['Price'].transform(lambda x: x.rolling(window=ma_period).mean())
        sd = self.data.groupby('Stock')['Price'].transform(lambda x: x.rolling(window=ma_period).std())
        self.data['Upper Band'] = self.data[f'{ma_period}MA'] + (sd * 2)
        self.data['Upper Mid Band'] = self.data[f'{ma_period}MA'] + (sd * 1)
        self.data['Lower Band'] = self.data[f'{ma_period}MA'] - (sd * 2)
        self.data['Lower Mid Band'] = self.data[f'{ma_period}MA'] - (sd * 1)

        self.whatToGraph.append('Bollinger Bands')

        return
    
    def maCalc(self, ma_period=21):
        self.data[f'{ma_period}MA'] = self.data.groupby('Stock')['Price'].transform(lambda x: x.rolling(window=ma_period).mean())
        
        self.whatToGraph.append(f'{ma_period}_MA')

        return

    def rsiCalc(self, window=14):
        # calculate the price differences
        price_diff = self.data.groupby('Stock')['Price'].diff()

        # seperate gains and losses
        gain = price_diff.where(price_diff > 0, 0)
        loss = -price_diff.where(price_diff < 0, 0)

        avg_gain = gain.groupby(self.data['Stock']).transform(lambda x: x.rolling(window=window).mean())
        avg_loss = loss.groupby(self.data['Stock']).transform(lambda x: x.rolling(window=window).mean())

        rs = avg_gain/avg_loss

        rsi = 100 - (100 / (1+rs))

        self.data[f'RSI {window}'] = rsi

        self.whatToGraph.append('RSI')

        return

    def stochRSICalc(self, window=14):
        rsi_min = self.data[f'RSI {window}'].groupby(self.data['Stock']).transform(lambda x: x.rolling(window=window).min())
        rsi_max = self.data[f'RSI {window}'].groupby(self.data['Stock']).transform(lambda x: x.rolling(window=window).max())
        stoch_rsi = (self.data[f'RSI {window}'] - rsi_min) / (rsi_max - rsi_min) * 100

        self.data[f'StochRSI {window}'] = stoch_rsi

        self.whatToGraph.append('StochRSI')
        return
    
    def macdCalc(self, slow_ema=26, fast_ema=12, signal=9):
        # long term ema
        self.data[f'{slow_ema}EMA'] = self.data.groupby('Stock')['Price'].transform(lambda x: x.ewm(span=slow_ema, adjust=False).mean())
        # short term ema
        self.data[f'{fast_ema}EMA'] = self.data.groupby('Stock')['Price'].transform(lambda x: x.ewm(span=fast_ema, adjust=False).mean())

        # calculate macd line
        self.data[f'MACD'] = self.data[f'{fast_ema}EMA'] - self.data[f'{slow_ema}EMA']

        self.data['MACD Signal'] = self.data.groupby('Stock')['Price'].transform(lambda x: x.ewm(span=signal, adjust=False).mean())

        self.whatToGraph.append('MACD')

        return
    
    def sdCalc(self, window=14):
        self.data['Daily Return'] = self.data.groupby('Stock')['Price'].pct_change()
        self.data[f'SD {window}'] = self.data.groupby('Stock')['Daily Return'].transform(lambda x: x.rolling(window=window).std())

        return
    
def loadPrices(fn):
    global nt, nInst
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt, nInst) = df.shape
    return (df.values).T

if __name__ == '__main__':
    #from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    #from statsmodels.tsa.stattools import acf, pacf
    # read the data from the text file
    file_path = './prices 750 days.txt'   
    prcAll = loadPrices(file_path)
    ma_period = 21
    ema_period = 9
    no_sd = 2

    df = ineffecieintStocks(prcAll)
    #rsi = df.rsiCalc()
    #dictionary = df.atrCalc()
    #print(dictionary)
    df.everythingGraph()
    """df = Stocks(prcAll)
    df.dailyReturnyCalc()
    df.atrCalc()
    stock = df.data[df.data['Stock']==1]
    yearly_mean = stock['ATR'].rolling(window=365).mean().iloc[-1]
    print(stock)
    print(yearly_mean)"""

    #print(vol)
    #print(df.macdCalc())

    #res = df.stochRSICalc(df.rsiCalc())
    #res = df.macdCalc()
    #df.bbGraph()
    #df.raw()
    #df.goldenCrossGraph()
    #df.rsiGraph()
    #df.stochRSIGraph()
    #df.macdGraph()
    #df.everythingGraph()
    # last_day = df.data.groupby('Stock').tail(1).reset_index(drop=True)

    
    #df.bbCalc()
    # Create directory if it doesn't exist
    

    """
    output_dir = Path("ACF and PACF")
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(50):
        df = Stocks(prcAll)
        df = df.data[df.data['Stock'] == i]
        df.set_index('Day', inplace=True)
        lags = 70
        optimal_lags_acf = []
        optimal_lags_pacf = []

        time_series = df.Price.diff().dropna()
        acf_vals = acf(time_series, nlags=lags)
        pacf_vals = pacf(time_series, nlags=lags)

        sig_acf_lags = np.where(np.abs(acf_vals) > 1.96/np.sqrt(len(df.Price)))[0]
        sig_pacf_lags = np.where(np.abs(pacf_vals) > 1.96/np.sqrt(len(df.Price)))[0]

        print(f"\n\nstock {i}")
        print(sig_acf_lags)
        print(sig_pacf_lags)

        # Use the maximum significant lag as the optimal lag
        if sig_acf_lags.size > 0:
            optimal_lags_acf.append(sig_acf_lags[-1])
        if sig_pacf_lags.size > 0:
            optimal_lags_pacf.append(sig_pacf_lags[-1])

    # Calculate the average optimal lags
    avg_optimal_lag_acf = np.median(optimal_lags_acf)
    avg_optimal_lag_pacf = np.median(optimal_lags_pacf)

    print(f'Average Optimal Lag (ACF): {avg_optimal_lag_acf}')
    print(f'Average Optimal Lag (PACF): {avg_optimal_lag_pacf}')
    """


    """
    # Create subplots for ACF
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    # ACF plot
    plot_acf(df.Price, lags=lags, ax=axes[0])
    axes[0].set_title('Autocorrelation Function (ACF)')
    # Differencing ACF plot
    plot_acf(df.Price.diff().dropna(), lags=lags, ax=axes[1])
    axes[1].set_title('(Differencing) Autocorrelation Function (ACF)')
    plt.tight_layout()
    # Save the plot to the directory
    plot_path = output_dir / f'stock{i}_ACF.png'
    plt.savefig(plot_path)
    plt.close()

    # Create subplots for PACF
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    # ACF plot
    plot_pacf(df.Price, lags=lags, ax=axes[0])
    axes[0].set_title('Partial Autocorrelation Function (PACF)')
    # Differencing ACF plot
    plot_pacf(df.Price.diff().dropna(), lags=lags, ax=axes[1])
    axes[1].set_title('(Differencing) Partial Autocorrelation Function (PACF)')
    plt.tight_layout()
    # Save the plot to the directory
    plot_path = output_dir / f'stock{i}_Partial_ACF.png'
    plt.savefig(plot_path)
    plt.close()


    # Create directory if it doesn't exist
    output_dir = Path("_testing_indicator_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    for id, bollinger_bands in res.items():
        output_file_path = output_dir / f'stock{id}.txt'
        bollinger_bands.to_csv(output_file_path, sep='\t',index=False)
    """
    