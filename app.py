import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass
import warnings
from datetime import datetime, timedelta
import os

warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Options Strategy Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import the backtester classes (copied from your notebook)
from backtester import (
    OptionType, PositionType, OptionLeg, OptionStrategy, 
    OptionStrategyTemplate, OptionsBacktester
)

# Load real crypto data
@st.cache_data
def load_data():
    """Load real crypto data for backtesting."""
    data_dir = "candle_data"
    
    # Available crypto datasets
    crypto_files = {
        "BTC-USDT": "BTC_USDT_1d.parquet",
        "ETH-USDT": "ETH_USDT_1d.parquet", 
        "PEPE-USDT": "PEPE_USDT_1d.parquet",
        "DOGE-USDT": "DOGE_USDT_1d.parquet"
    }
    
    datasets = {}
    
    for crypto, filename in crypto_files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_parquet(filepath)
            # Ensure timestamp is the index
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            elif df.index.name != 'timestamp':
                df.index.name = 'timestamp'
            datasets[crypto] = df
        else:
            st.warning(f"File not found: {filepath}")
    
    return datasets

def create_strategy_diagram(strategy_name):
    """Create a simple visualization of the option strategy."""
    fig = go.Figure()
    
    # Strategy payoff diagrams (simplified)
    spot_range = np.linspace(0.8, 1.2, 100)
    
    if strategy_name == "Covered Call":
        # Long stock + short call
        stock_payoff = spot_range - 1.0
        call_payoff = np.minimum(0, 1.05 - spot_range)  # Short call at 105% strike
        total_payoff = stock_payoff + call_payoff + 0.02  # Plus premium
        
        # Color based on positive/negative
        colors = ['green' if p >= 0 else 'red' for p in total_payoff]
        
        fig.add_trace(go.Scatter(x=spot_range*100, y=total_payoff*100, 
                                name='Total P&L', 
                                line=dict(width=3, color='blue'),
                                marker=dict(color=colors, size=2)))
        
    elif strategy_name == "Cash Secured Put":
        put_payoff = np.minimum(0, spot_range - 0.95) + 0.02  # Short put + premium
        colors = ['green' if p >= 0 else 'red' for p in put_payoff]
        
        fig.add_trace(go.Scatter(x=spot_range*100, y=put_payoff*100, 
                                name='Cash Secured Put', 
                                line=dict(width=3, color='blue'),
                                marker=dict(color=colors, size=2)))
        
    elif strategy_name == "Long Straddle":
        call_payoff = np.maximum(0, spot_range - 1.0) - 0.03
        put_payoff = np.maximum(0, 1.0 - spot_range) - 0.03
        total_payoff = call_payoff + put_payoff
        colors = ['green' if p >= 0 else 'red' for p in total_payoff]
        
        fig.add_trace(go.Scatter(x=spot_range*100, y=total_payoff*100, 
                                name='Long Straddle', 
                                line=dict(width=3, color='blue'),
                                marker=dict(color=colors, size=2)))
        
    elif strategy_name == "Iron Condor":
        # Simplified iron condor
        payoff = np.where(spot_range < 0.95, -(spot_range - 0.95) + 0.01,
                 np.where(spot_range > 1.05, -(spot_range - 1.05) + 0.01, 0.01))
        colors = ['green' if p >= 0 else 'red' for p in payoff]
        
        fig.add_trace(go.Scatter(x=spot_range*100, y=payoff*100, 
                                name='Iron Condor', 
                                line=dict(width=3, color='blue'),
                                marker=dict(color=colors, size=2)))
        
    elif strategy_name == "Bull Call Spread":
        long_call = np.maximum(0, spot_range - 1.0) - 0.03
        short_call = -(np.maximum(0, spot_range - 1.05) - 0.01)
        total_payoff = long_call + short_call
        colors = ['green' if p >= 0 else 'red' for p in total_payoff]
        
        fig.add_trace(go.Scatter(x=spot_range*100, y=total_payoff*100, 
                                name='Bull Call Spread', 
                                line=dict(width=3, color='blue'),
                                marker=dict(color=colors, size=2)))
        
    elif strategy_name == "Long Strangle":
        call_payoff = np.maximum(0, spot_range - 1.05) - 0.015
        put_payoff = np.maximum(0, 0.95 - spot_range) - 0.015
        total_payoff = call_payoff + put_payoff
        colors = ['green' if p >= 0 else 'red' for p in total_payoff]
        
        fig.add_trace(go.Scatter(x=spot_range*100, y=total_payoff*100, 
                                name='Long Strangle', 
                                line=dict(width=3, color='blue'),
                                marker=dict(color=colors, size=2)))
    
    fig.update_layout(
        title=f"{strategy_name} Payoff Diagram",
        xaxis_title="Spot Price (%)",
        yaxis_title="P&L (%)",
        height=300,
        showlegend=True
    )
    fig.add_hline(y=0, line_color="gray", opacity=0.5)
    fig.add_vline(x=100, line_color="gray", opacity=0.5, line_dash="dash")
    
    return fig

def format_stats_table(stats_str):
    """Convert the stats string to a nicely formatted dataframe."""
    lines = stats_str.split('\n')
    data = []
    
    for line in lines:
        if '|' in line and 'Metric' not in line and '---' not in line:
            parts = [part.strip() for part in line.split('|') if part.strip()]
            if len(parts) >= 2:
                data.append({'Metric': parts[0], 'Value': parts[1]})
    
    if data:
        df = pd.DataFrame(data)
        return df
    else:
        # Fallback parsing
        stats_dict = {}
        for line in lines:
            if ':' in line and not line.startswith('+') and not line.startswith('|'):
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    stats_dict[key] = value
        
        df = pd.DataFrame(list(stats_dict.items()), columns=['Metric', 'Value'])
        return df

def main():
    st.title("📈 Options Strategy Backtester")
    st.markdown("Backtest various options strategies on real cryptocurrency data with customizable parameters.")
    
    # Sidebar for strategy selection and parameters
    st.sidebar.header("Strategy Configuration")
    
    # Load datasets
    datasets = load_data()
    
    if not datasets:
        st.error("No data files found! Please ensure candle_data folder exists with parquet files.")
        return
    
    # Crypto selection
    selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", list(datasets.keys()))
    
    # Strategy selection
    strategy_options = [
        "Covered Call",
        "Cash Secured Put", 
        "Long Straddle",
        "Iron Condor",
        "Bull Call Spread",
        "Long Strangle"
    ]
    
    selected_strategy = st.sidebar.selectbox("Select Strategy", strategy_options)
    
    # Show strategy diagram
    st.sidebar.subheader("Strategy Visualization")
    strategy_fig = create_strategy_diagram(selected_strategy)
    st.sidebar.plotly_chart(strategy_fig, use_container_width=True)
    
    # Strategy parameters
    st.sidebar.subheader("Strategy Parameters")
    
    # Common parameters with number inputs instead of sliders
    if selected_strategy == "Covered Call":
        strike_pct = st.sidebar.number_input("Strike Price (%)", min_value=100, max_value=120, value=105, step=1) / 100
        premium_pct = st.sidebar.number_input("Premium (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.1) / 100
        underlying_position = st.sidebar.number_input("Underlying Position", min_value=0, max_value=1000, value=1, step=1)
        underlying_held = st.sidebar.number_input("Already Held", min_value=0, max_value=1000, value=1, step=1)
        underlying_avg_cost = st.sidebar.number_input("Average Cost ($)", min_value=0.0, max_value=100000.0, value=30000.0, step=100.0)
        
        strategy = OptionStrategyTemplate.covered_call(
            strike_pct=strike_pct,
            premium_pct=premium_pct,
            underlying_position=underlying_position,
            underlying_held=underlying_held,
            underlying_avg_cost=underlying_avg_cost
        )
        
    elif selected_strategy == "Cash Secured Put":
        strike_pct = st.sidebar.number_input("Strike Price (%)", min_value=80, max_value=100, value=95, step=1) / 100
        premium_pct = st.sidebar.number_input("Premium (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.1) / 100
        
        strategy = OptionStrategyTemplate.cash_secured_put(
            strike_pct=strike_pct,
            premium_pct=premium_pct
        )
        
    elif selected_strategy == "Long Straddle":
        strike_pct = st.sidebar.number_input("Strike Price (%)", min_value=95, max_value=105, value=100, step=1) / 100
        call_premium = st.sidebar.number_input("Call Premium (%)", min_value=1.0, max_value=6.0, value=3.0, step=0.1) / 100
        put_premium = st.sidebar.number_input("Put Premium (%)", min_value=1.0, max_value=6.0, value=3.0, step=0.1) / 100
        
        strategy = OptionStrategyTemplate.long_straddle(
            strike_pct=strike_pct,
            call_premium=call_premium,
            put_premium=put_premium
        )
        
    elif selected_strategy == "Iron Condor":
        put_short = st.sidebar.number_input("Put Short Strike (%)", min_value=85, max_value=100, value=95, step=1) / 100
        put_long = st.sidebar.number_input("Put Long Strike (%)", min_value=80, max_value=95, value=90, step=1) / 100
        call_short = st.sidebar.number_input("Call Short Strike (%)", min_value=100, max_value=115, value=105, step=1) / 100
        call_long = st.sidebar.number_input("Call Long Strike (%)", min_value=105, max_value=120, value=110, step=1) / 100
        short_premium = st.sidebar.number_input("Short Premium (%)", min_value=0.5, max_value=4.0, value=2.0, step=0.1) / 100
        long_premium = st.sidebar.number_input("Long Premium (%)", min_value=0.2, max_value=2.0, value=1.0, step=0.1) / 100
        
        strategy = OptionStrategyTemplate.iron_condor(
            put_short=put_short,
            put_long=put_long,
            call_short=call_short,
            call_long=call_long,
            short_premium=short_premium,
            long_premium=long_premium
        )
        
    elif selected_strategy == "Bull Call Spread":
        long_strike = st.sidebar.number_input("Long Strike (%)", min_value=95, max_value=105, value=100, step=1) / 100
        short_strike = st.sidebar.number_input("Short Strike (%)", min_value=100, max_value=115, value=105, step=1) / 100
        long_premium = st.sidebar.number_input("Long Premium (%)", min_value=1.0, max_value=5.0, value=3.0, step=0.1) / 100
        short_premium = st.sidebar.number_input("Short Premium (%)", min_value=0.2, max_value=3.0, value=1.0, step=0.1) / 100
        
        strategy = OptionStrategyTemplate.bull_call_spread(
            long_strike=long_strike,
            short_strike=short_strike,
            long_premium=long_premium,
            short_premium=short_premium
        )
        
    elif selected_strategy == "Long Strangle":
        call_strike = st.sidebar.number_input("Call Strike (%)", min_value=100, max_value=115, value=105, step=1) / 100
        put_strike = st.sidebar.number_input("Put Strike (%)", min_value=85, max_value=100, value=95, step=1) / 100
        call_premium = st.sidebar.number_input("Call Premium (%)", min_value=0.5, max_value=3.0, value=1.5, step=0.1) / 100
        put_premium = st.sidebar.number_input("Put Premium (%)", min_value=0.5, max_value=3.0, value=1.5, step=0.1) / 100
        
        strategy = OptionStrategyTemplate.long_strangle(
            call_strike=call_strike,
            put_strike=put_strike,
            call_premium=call_premium,
            put_premium=put_premium
        )
    
    # Backtesting parameters
    st.sidebar.subheader("Backtesting Parameters")
    expiry_days = st.sidebar.number_input("Expiry Days", min_value=1, max_value=30, value=7, step=1)
    
    # Date range
    price_data = datasets[selected_crypto]
    available_start = price_data.index.min().date()
    available_end = price_data.index.max().date()
    
    default_start = max(available_start, datetime(2024, 1, 1).date())
    default_end = min(available_end, datetime(2024, 12, 31).date())
    
    start_date = st.sidebar.date_input("Start Date", default_start, min_value=available_start, max_value=available_end)
    end_date = st.sidebar.date_input("End Date", default_end, min_value=available_start, max_value=available_end)
    
    trade_frequency = st.sidebar.selectbox(
        "Trade Frequency", 
        ["non_overlapping", "daily", "weekly", "monthly"]
    )
    
    if trade_frequency in ["weekly", "monthly"]:
        entry_day = st.sidebar.selectbox(
            "Entry Day of Week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
        entry_day_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(entry_day)
    else:
        entry_day_of_week = 0
    
    # Load data and run backtest
    if st.sidebar.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            # Use selected crypto data
            backtester = OptionsBacktester(price_data)
            
            # Show data info
            st.info(f"Using {selected_crypto} data: {len(price_data)} days from {price_data.index[0].date()} to {price_data.index[-1].date()}")
            
            # Run backtest
            stats_str = backtester.summary_stats(
                strategy=strategy,
                expiry_days=expiry_days,
                start_date=str(start_date),
                end_date=str(end_date),
                trade_frequency=trade_frequency,
                entry_day_of_week=entry_day_of_week
            )
            
            fig = backtester.plot_backtest_results(
                strategy=strategy,
                expiry_days=expiry_days,
                start_date=str(start_date),
                end_date=str(end_date),
                trade_frequency=trade_frequency,
                entry_day_of_week=entry_day_of_week
            )
            
            # Store results in session state
            st.session_state.stats_str = stats_str
            st.session_state.fig = fig
            st.session_state.strategy_name = selected_strategy
            st.session_state.crypto_name = selected_crypto
    
    # Display results
    if hasattr(st.session_state, 'stats_str'):
        st.header(f"📊 {st.session_state.strategy_name} - {st.session_state.crypto_name} Backtest Results")
        
        # Create two columns for layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Performance Summary")
            
            # Format and display stats table
            stats_df = format_stats_table(st.session_state.stats_str)
            if not stats_df.empty:
                # Style the dataframe
                styled_df = stats_df.style.set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#f0f2f6'), ('font-weight', 'bold')]},
                    {'selector': 'td', 'props': [('text-align', 'left')]},
                ])
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.text(st.session_state.stats_str)
        
        with col2:
            st.subheader("Key Metrics")
            
            # Extract key metrics for display
            if not stats_df.empty:
                metrics_dict = dict(zip(stats_df['Metric'], stats_df['Value']))
                
                # Create metric cards
                metric_cols = st.columns(3)
                
                with metric_cols[0]:
                    if 'total_return' in metrics_dict:
                        st.metric("Total Return", f"{metrics_dict['total_return']}%")
                    if 'win_rate' in metrics_dict:
                        st.metric("Win Rate", f"{metrics_dict['win_rate']}%")
                
                with metric_cols[1]:
                    if 'total_pnl' in metrics_dict:
                        st.metric("Total P&L", f"${metrics_dict['total_pnl']}")
                    if 'sharpe_ratio' in metrics_dict:
                        st.metric("Sharpe Ratio", metrics_dict['sharpe_ratio'])
                
                with metric_cols[2]:
                    if 'total_trades' in metrics_dict:
                        st.metric("Total Trades", metrics_dict['total_trades'])
                    if 'max_drawdown' in metrics_dict:
                        st.metric("Max Drawdown", f"${metrics_dict['max_drawdown']}")
        
        # Display the backtest chart
        st.subheader("Detailed Analysis")
        if hasattr(st.session_state, 'fig'):
            st.plotly_chart(st.session_state.fig, use_container_width=True)
    
    else:
        st.info("👈 Configure your strategy parameters and click 'Run Backtest' to see results.")
        
        # Show dataset information
        if datasets:
            st.subheader("📊 Available Datasets")
            for crypto, data in datasets.items():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"{crypto}", f"{len(data)} days")
                with col2:
                    st.metric("Date Range", f"{data.index[0].date()} to {data.index[-1].date()}")
                with col3:
                    st.metric("Current Price", f"${data['close'].iloc[-1]:.2f}")
        
        # Show some educational content
        st.header("📚 About Options Strategies")
        
        with st.expander("Covered Call"):
            st.write("""
            **Strategy**: Long stock + Short call option
            **Market Outlook**: Neutral to slightly bullish
            **Profit**: Limited (stock appreciation up to strike + premium collected)
            **Loss**: Unlimited downside protection only by premium collected
            **Best When**: You own the stock and expect sideways/modest upward movement
            """)
        
        with st.expander("Cash Secured Put"):
            st.write("""
            **Strategy**: Short put option with cash to buy stock if assigned
            **Market Outlook**: Neutral to bullish
            **Profit**: Limited to premium collected
            **Loss**: Stock price minus strike minus premium
            **Best When**: You want to buy stock at a lower price and collect premium while waiting
            """)
        
        with st.expander("Long Straddle"):
            st.write("""
            **Strategy**: Long call + Long put at same strike
            **Market Outlook**: High volatility expected (direction unknown)
            **Profit**: Unlimited in either direction
            **Loss**: Limited to total premium paid
            **Best When**: Expecting big price movement but unsure of direction
            """)

if __name__ == "__main__":
    main()
