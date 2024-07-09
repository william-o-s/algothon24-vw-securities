import numpy as np
# Combo of EMA and Mean reversion, performed VERY BAD

def EMA(prices, window):
    alpha = 2 / (window + 1)
    emas = np.zeros_like(prices)
    emas[:, :window] = prices[:, :window]  # Initialize first 'window' days

    for t in range(window, prices.shape[1]):
        emas[:, t] = alpha * prices[:, t] + (1 - alpha) * emas[:, t-1]

    return emas

def mean_reversion_signal(prices, lower_percentile, upper_percentile):
    nInst, nt = prices.shape
    signals = np.zeros(nInst)
    
    # Calculate mean reversion thresholds
    lower_threshold = np.percentile(prices, lower_percentile, axis=1)
    upper_threshold = np.percentile(prices, upper_percentile, axis=1)
    
    current_prices = prices[:, -1]
    
    buy_signals = current_prices < lower_threshold
    sell_signals = current_prices > upper_threshold
    signals = buy_signals.astype(int) - sell_signals.astype(int)
    
    return signals

def getMyPosition(prices, short_window=3, long_window=94, lower_percentile=5, upper_percentile=90, risk_per_trade=0.01, account_size=100000):
    nInst, nt = prices.shape
    positions = np.zeros(nInst)

    if nt >= long_window:  # Make sure there is enough data
        ema_short = EMA(prices, short_window)[:, -1]  # Last value of the short-term EMA
        ema_long = EMA(prices, long_window)[:, -1]  # Last value of the long-term EMA

        # Generate EMA crossover signals
        buy_signals_ema = ema_short > ema_long
        sell_signals_ema = ema_short < ema_long
        signals_ema = buy_signals_ema.astype(int) - sell_signals_ema.astype(int)

        # Generate mean reversion signals
        signals_mean_reversion = mean_reversion_signal(prices, lower_percentile, upper_percentile)

        # Combine signals (e.g., take an average or use more sophisticated logic)
        combined_signals = (signals_ema + signals_mean_reversion) / 2
        combined_signals = np.round(combined_signals).astype(int)  # Convert to discrete signals

        # Calculate position size based on risk per trade
        latest_prices = prices[:, -1]
        stop_loss_distance = np.abs(ema_short - ema_long)  # Example stop-loss distance
        dollar_risk_per_trade = risk_per_trade * account_size
        position_size = dollar_risk_per_trade / stop_loss_distance
        max_pos_size = np.minimum(position_size, (10000 / latest_prices)).astype(int)

        # Apply combined signals and adjust for position limits
        positions = combined_signals * max_pos_size
        positions = np.clip(positions, -max_pos_size, max_pos_size)

    return positions
