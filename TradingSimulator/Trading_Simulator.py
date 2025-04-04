# step1: pip install pandas numpy plotly
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Input User for CSV File ---
print("Please enter the path to your CSV file (e.g., TH2027.csv or /full/path/to/TH2027.csv):")
input_file = input().strip()  # Get file path from user, remove extra whitespace

try:
    # Load the CSV file
    df = pd.read_csv(input_file, parse_dates=['Date'], thousands=',')
except FileNotFoundError:
    print(f"Error: File '{input_file}' not found. Please check the path and try again.")
    exit(1)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# Ensure columns are correctly named and typed
df['Date'] = pd.to_datetime(df['Date'])
df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].astype(float)
df['Volume'] = df['Volume'].astype(int)

# --- Calculate Technical Indicators ---
df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
df['VWMA20'] = (df['Close'] * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()


def calculate_rsi(data, periods=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    avg_gain = gain.rolling(window=periods, min_periods=periods).mean()
    avg_loss = loss.rolling(window=periods, min_periods=periods).mean()
    for i in range(periods, len(data)):
        if i == periods:
            continue
        avg_gain.iloc[i] = ((avg_gain.iloc[i - 1] * (periods - 1)) + gain.iloc[i]) / periods
        avg_loss.iloc[i] = ((avg_loss.iloc[i - 1] * (periods - 1)) + loss.iloc[i]) / periods
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


df['RSI'] = calculate_rsi(df)
df = df.dropna().reset_index(drop=True)

# --- Trading Simulation ---
initial_capital = 10000
transaction_fee = 2
stop_loss_percent = 0.05
take_profit_percent = 0.10

cash = initial_capital
shares = 0
trades = []
position = None

for i in range(1, len(df)):
    if (df['EMA10'].iloc[i - 1] <= df['VWMA20'].iloc[i - 1] and
            df['EMA10'].iloc[i] > df['VWMA20'].iloc[i] and
            df['RSI'].iloc[i] < 30 and
            shares == 0):
        entry_price = df['Close'].iloc[i]
        shares = (cash - transaction_fee) // entry_price
        cash -= (shares * entry_price + transaction_fee)
        position = {
            'entry_time': df['Date'].iloc[i],
            'entry_price': entry_price,
            'shares': shares,
            'stop_loss': entry_price * (1 - stop_loss_percent),
            'take_profit': entry_price * (1 + take_profit_percent)
        }

    if shares > 0 and position:
        current_price = df['Close'].iloc[i]
        if current_price <= position['stop_loss']:
            exit_price = position['stop_loss']
            cash += shares * exit_price - transaction_fee
            profit_loss = (exit_price - position['entry_price']) * shares - 2 * transaction_fee
            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': df['Date'].iloc[i],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'profit_loss': profit_loss,
                'stop_loss': position['stop_loss'],
                'take_profit': position['take_profit'],
                'exit_reason': 'stop_loss'
            })
            shares = 0
            position = None
        elif current_price >= position['take_profit']:
            exit_price = position['take_profit']
            cash += shares * exit_price - transaction_fee
            profit_loss = (exit_price - position['entry_price']) * shares - 2 * transaction_fee
            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': df['Date'].iloc[i],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'profit_loss': profit_loss,
                'stop_loss': position['stop_loss'],
                'take_profit': position['take_profit'],
                'exit_reason': 'take_profit'
            })
            shares = 0
            position = None
        elif (df['EMA10'].iloc[i - 1] >= df['EMA20'].iloc[i - 1] and
              df['EMA10'].iloc[i] < df['EMA20'].iloc[i] and
              df['RSI'].iloc[i] > 70):
            exit_price = df['Close'].iloc[i]
            cash += shares * exit_price - transaction_fee
            profit_loss = (exit_price - position['entry_price']) * shares - 2 * transaction_fee
            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': df['Date'].iloc[i],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'profit_loss': profit_loss,
                'stop_loss': position['stop_loss'],
                'take_profit': position['take_profit'],
                'exit_reason': 'sell_signal'
            })
            shares = 0
            position = None

trade_df = pd.DataFrame(trades)

# --- Statistics ---
total_trades = len(trade_df)
total_profit_loss = trade_df['profit_loss'].sum() if not trade_df.empty else 0
profit_positions = len(trade_df[trade_df['profit_loss'] > 0]) if not trade_df.empty else 0
loss_positions = len(trade_df[trade_df['profit_loss'] < 0]) if not trade_df.empty else 0
avg_profit = trade_df[trade_df['profit_loss'] > 0]['profit_loss'].mean() if profit_positions > 0 else 0
avg_loss = abs(trade_df[trade_df['profit_loss'] < 0]['profit_loss'].mean()) if loss_positions > 0 else 0
p_to_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else float('inf')

print("Trade Details:")
if not trade_df.empty:
    print(trade_df[['entry_time', 'exit_time', 'entry_price', 'exit_price', 'profit_loss',
                    'stop_loss', 'take_profit', 'exit_reason']].to_string(index=False))
else:
    print("No trades executed.")
print("\nOverall Statistics:")
print(f"Total Trades: {total_trades}")
print(f"Total Profit/Loss: ${total_profit_loss:.2f}")
print(f"Profit Positions: {profit_positions}")
print(f"Loss Positions: {loss_positions}")
print(f"Profit-to-Loss Ratio: {p_to_loss_ratio:.2f}")

# --- Visualization ---
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df['Date'],
                             open=df['Open'],
                             high=df['High'],
                             low=df['Low'],
                             close=df['Close'],
                             name='Candlestick'))
fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA10'], name='EMA10', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA20'], name='EMA20', line=dict(color='orange')))
fig.add_trace(go.Scatter(x=df['Date'], y=df['VWMA20'], name='VWMA20', line=dict(color='green')))

for _, trade in trade_df.iterrows():
    fig.add_vline(x=trade['entry_time'], line_dash="dash", line_color="green")
    fig.add_annotation(x=trade['entry_time'], y=trade['entry_price'], text="Buy", showarrow=True, arrowhead=1)
    fig.add_vline(x=trade['exit_time'], line_dash="dash", line_color="red")
    fig.add_annotation(x=trade['exit_time'], y=trade['exit_price'], text="Sell", showarrow=True, arrowhead=1)

fig.update_layout(
    title='Trading Simulation',  # Fixed: Removed undefined 'ticker'
    yaxis_title='Price',
    xaxis_title='Date',
    template='plotly_dark',
    showlegend=True
)
fig.show()

final_value = cash + (shares * df['Close'].iloc[-1] if shares > 0 else 0)
print(f"Final Portfolio Value: ${final_value:.2f}")