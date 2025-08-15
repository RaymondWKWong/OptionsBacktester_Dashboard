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
            datasets[crypto] = df
        else:
            st.warning(f"File not found: {filepath}")
    
    return datasets

def create_strategy_diagram(strategy_name):
    """Create a simple visualisation of the option strategy."""
    fig = go.Figure()
    # Strategy payoff diagrams
    spot_range = np.linspace(0.8, 1.2, 100)
    
    def add_colored_line(x_data, y_data):
        """Add line segments colored by positive/negative values"""
        # Create continuous line by duplicating zero crossings
        x_extended = []
        y_extended = []
        colors = []
        
        for i in range(len(y_data)):
            x_extended.append(x_data[i])
            y_extended.append(y_data[i])
            colors.append('green' if y_data[i] >= 0 else 'red')
            
            # Check for zero crossing
            if i < len(y_data) - 1:
                if (y_data[i] >= 0 and y_data[i+1] < 0) or (y_data[i] < 0 and y_data[i+1] >= 0):
                    # Interpolate zero crossing point
                    zero_x = x_data[i] + (x_data[i+1] - x_data[i]) * (-y_data[i] / (y_data[i+1] - y_data[i]))
                    x_extended.append(zero_x)
                    y_extended.append(0)
                    colors.append('green' if y_data[i+1] >= 0 else 'red')
        
        # Split into segments by color
        current_color = colors[0]
        segment_x = [x_extended[0]]
        segment_y = [y_extended[0]]
        
        for i in range(1, len(colors)):
            if colors[i] == current_color:
                segment_x.append(x_extended[i])
                segment_y.append(y_extended[i])
            else:
                # End current segment and start new one
                fig.add_trace(go.Scatter(x=segment_x, y=segment_y,
                                        mode='lines',
                                        line=dict(width=3, color=current_color),
                                        showlegend=False))
                current_color = colors[i]
                segment_x = [x_extended[i-1], x_extended[i]]  # Include transition point
                segment_y = [y_extended[i-1], y_extended[i]]
        
        # Add final segment
        fig.add_trace(go.Scatter(x=segment_x, y=segment_y,
                                mode='lines',
                                line=dict(width=3, color=current_color),
                                showlegend=False))
    
    if strategy_name == "Covered Call":
        # Long stock + short call
        stock_payoff = spot_range - 1.0
        call_payoff = np.minimum(0, 1.05 - spot_range)  # Short call at 105% strike
        total_payoff = stock_payoff + call_payoff + 0.02  # Plus premium
        add_colored_line(spot_range*100, total_payoff*100)
    
    elif strategy_name == "Cash Secured Put":
        put_payoff = np.minimum(0, spot_range - 0.95) + 0.02  # Short put + premium
        add_colored_line(spot_range*100, put_payoff*100)
    
    elif strategy_name == "Long Straddle":
        call_payoff = np.maximum(0, spot_range - 1.0) - 0.03
        put_payoff = np.maximum(0, 1.0 - spot_range) - 0.03
        total_payoff = call_payoff + put_payoff
        add_colored_line(spot_range*100, total_payoff*100)
    
    elif strategy_name == "Iron Condor":
        # Simplified iron condor
        payoff = np.where(spot_range < 0.95, -(spot_range - 0.95) + 0.01,
                         np.where(spot_range > 1.05, -(spot_range - 1.05) + 0.01, 0.01))
        add_colored_line(spot_range*100, payoff*100)
    
    elif strategy_name == "Bull Call Spread":
        long_call = np.maximum(0, spot_range - 1.0) - 0.03
        short_call = -(np.maximum(0, spot_range - 1.05) - 0.01)
        total_payoff = long_call + short_call
        add_colored_line(spot_range*100, total_payoff*100)
    
    elif strategy_name == "Long Strangle":
        call_payoff = np.maximum(0, spot_range - 1.05) - 0.015
        put_payoff = np.maximum(0, 0.95 - spot_range) - 0.015
        total_payoff = call_payoff + put_payoff
        add_colored_line(spot_range*100, total_payoff*100)
    
    fig.update_layout(
        title=f"{strategy_name} Payoff Diagram",
        xaxis_title="",
        yaxis_title="",
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        height=300,
        showlegend=False
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

def safe_summary_stats(backtester, **kwargs):
    """Wrapper for summary_stats with error handling."""
    try:
        return backtester.summary_stats(**kwargs)
    except IndexError as e:
        if "index 0 is out of bounds" in str(e):
            # Return a basic summary when detailed analysis fails
            try:
                # Create a copy of kwargs without entry_day_of_week for backtest_strategy
                backtest_kwargs = kwargs.copy()
                if 'entry_day_of_week' in backtest_kwargs:
                    del backtest_kwargs['entry_day_of_week']
                
                # Run basic backtest to get trade count
                daily_data, weekday_data = backtester.backtest_strategy(**backtest_kwargs)
                
                if daily_data.empty:
                    return "No trades were generated for the selected parameters. Try adjusting the date range or strategy parameters."
                else:
                    # Basic stats without day-of-week analysis
                    total_trades = len(daily_data)
                    
                    # Check if we have the required columns
                    if 'current_pnl' in daily_data.columns:
                        total_pnl = daily_data['current_pnl'].sum()
                        win_rate = (daily_data['current_pnl'] > 0).mean() * 100
                    else:
                        total_pnl = 0
                        win_rate = 0
                    
                    return f"""Basic Results:
Total Trades: {total_trades}
Total P&L: ${total_pnl:.2f}
Win Rate: {win_rate:.1f}%

Note: Detailed analysis failed due to insufficient data for some metrics.
Try a longer date range or different parameters for full analysis."""
            except Exception as e2:
                return f"Error running backtest: {str(e2)}"
        else:
            return f"Error in analysis: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

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
    strategy_fig = create_strategy_diagram(selected_strategy)
    st.sidebar.plotly_chart(strategy_fig, use_container_width=True)
    
    # Strategy parameters
    st.sidebar.subheader("Strategy Parameters")
    
    # Common parameters with number inputs instead of sliders
    if selected_strategy == "Covered Call":
        strike_pct = st.sidebar.number_input("Strike Price (%)", min_value=100, max_value=120, value=105, step=1) / 100
        underlying_position = st.sidebar.number_input("Underlying Position", min_value=0, max_value=1000, value=1, step=1)
        underlying_held = st.sidebar.number_input("Already Held", min_value=0, max_value=1000, value=1, step=1)
        underlying_avg_cost = st.sidebar.number_input("Average Cost ($)", min_value=0.0, max_value=200000.0, value=45000.0, step=100.0) if underlying_held > 0 else None
        
        strategy = OptionStrategyTemplate.covered_call(
            strike_pct=strike_pct,
            underlying_position=underlying_position,
            underlying_held=underlying_held,
            underlying_avg_cost=underlying_avg_cost
        )
        
    elif selected_strategy == "Cash Secured Put":
        strike_pct = st.sidebar.number_input("Strike Price (%)", min_value=80, max_value=100, value=95, step=1) / 100
        
        strategy = OptionStrategyTemplate.cash_secured_put(
            strike_pct=strike_pct
        )
        
    elif selected_strategy == "Long Straddle":
        strike_pct = st.sidebar.number_input("Strike Price (%)", min_value=95, max_value=105, value=100, step=1) / 100
        
        strategy = OptionStrategyTemplate.long_straddle(
            strike_pct=strike_pct
        )
        
    elif selected_strategy == "Iron Condor":
        put_short = st.sidebar.number_input("Put Short Strike (%)", min_value=85, max_value=100, value=95, step=1) / 100
        put_long = st.sidebar.number_input("Put Long Strike (%)", min_value=80, max_value=95, value=90, step=1) / 100
        call_short = st.sidebar.number_input("Call Short Strike (%)", min_value=100, max_value=115, value=105, step=1) / 100
        call_long = st.sidebar.number_input("Call Long Strike (%)", min_value=105, max_value=120, value=110, step=1) / 100
        
        strategy = OptionStrategyTemplate.iron_condor(
            put_short=put_short,
            put_long=put_long,
            call_short=call_short,
            call_long=call_long
        )
        
    elif selected_strategy == "Bull Call Spread":
        long_strike = st.sidebar.number_input("Long Strike (%)", min_value=95, max_value=105, value=100, step=1) / 100
        short_strike = st.sidebar.number_input("Short Strike (%)", min_value=100, max_value=115, value=105, step=1) / 100
        
        strategy = OptionStrategyTemplate.bull_call_spread(
            long_strike=long_strike,
            short_strike=short_strike
        )
        
    elif selected_strategy == "Long Strangle":
        call_strike = st.sidebar.number_input("Call Strike (%)", min_value=100, max_value=115, value=105, step=1) / 100
        put_strike = st.sidebar.number_input("Put Strike (%)", min_value=85, max_value=100, value=95, step=1) / 100
        
        strategy = OptionStrategyTemplate.long_strangle(
            call_strike=call_strike,
            put_strike=put_strike
        )
    
    # Backtesting parameters
    st.sidebar.subheader("Backtesting Parameters")
    expiry_days = st.sidebar.number_input("Expiry Days", min_value=1, max_value=30, value=7, step=1)
    interest_rate = st.sidebar.number_input("Interest Rate", min_value=0.0, max_value=0.10, value=0.05, step=0.01)
    volatility = st.sidebar.number_input("Volatility", min_value=0.1, max_value=1.0, value=0.50, step=0.05)
    
    # Date range
    price_data = datasets[selected_crypto]
    if not isinstance(price_data.index, pd.DatetimeIndex):  # Ensure index is datetime
        price_data.index = pd.to_datetime(price_data.index)
    available_start = price_data.index.min().date()
    available_end = price_data.index.max().date()
    
    default_start = max(available_start, datetime(2024, 7, 16).date())
    default_end = min(available_end, datetime(2025, 7, 15).date())

    # Ensure all are datetime.date
    if isinstance(available_start, pd.Timestamp):
        available_start = available_start.date()
    if isinstance(available_end, pd.Timestamp):
        available_end = available_end.date()
    if isinstance(default_start, pd.Timestamp):
        default_start = default_start.date()
    if isinstance(default_end, pd.Timestamp):
        default_end = default_end.date()

    start_date = st.sidebar.date_input("Start Date", default_start, min_value=available_start, max_value=available_end)
    end_date = st.sidebar.date_input("End Date", default_end, min_value=available_start, max_value=available_end)
    
    trade_frequency = st.sidebar.selectbox(
        "Trade Frequency", 
        ["non_overlapping", "daily", "weekly", "monthly"]
    )
    
    # Entry day of week selection (always available, not just for weekly/monthly)
    entry_day_option = st.sidebar.selectbox(
        "Entry Day of Week Filter",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        help="Filter results to only show trades entered on specific days of the week. This affects the analysis and statistics."
    )
    
    if entry_day_option == "All Days":
        entry_day_of_week = None
    else:
        entry_day_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(entry_day_option) + 1
    
    # Load data and run backtest
    if st.sidebar.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            # Use selected crypto data
            backtester = OptionsBacktester(price_data)
            
            # Show data info
            data_info = f"Using {selected_crypto} data: {len(price_data)} days from {price_data.index[0].date()} to {price_data.index[-1].date()}"
            if entry_day_of_week:
                day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][entry_day_of_week - 1]
                data_info += f" (filtered for {day_name}s only)"
            st.info(data_info)
            
            # Parameters for backtesting
            backtest_params = {
                'strategy': strategy,
                'expiry_days': expiry_days,
                'start_date': str(start_date),
                'end_date': str(end_date),
                'trade_frequency': trade_frequency,
                'entry_day_of_week': entry_day_of_week,
                'interest_rate': interest_rate,
                'volatility': volatility
            }
            
            # Run backtest with safe wrapper
            stats_str = safe_summary_stats(backtester, **backtest_params)
            
            # Get detailed backtest data for tables
            try:
                backtest_kwargs = backtest_params.copy()
                if 'entry_day_of_week' in backtest_kwargs:
                    del backtest_kwargs['entry_day_of_week']  # Remove for backtest_strategy call
                daily_df, weekday_df = backtester.backtest_strategy(**backtest_kwargs)
            except Exception as e:
                st.warning(f"Could not get detailed backtest data: {str(e)}")
                daily_df, weekday_df = None, None
            
            # Try to get the plot
            try:
                fig = backtester.plot_backtest_results(**backtest_params)
            except Exception as e:
                st.warning(f"Could not generate plot: {str(e)}")
                fig = None
            
            # Store results in session state
            st.session_state.stats_str = stats_str
            if fig is not None:
                st.session_state.fig = fig
            if daily_df is not None:
                st.session_state.daily_df = daily_df
            if weekday_df is not None:
                st.session_state.weekday_df = weekday_df
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
        if hasattr(st.session_state, 'fig') and st.session_state.fig is not None:
            st.plotly_chart(st.session_state.fig, use_container_width=True)
        else:
            st.warning("No chart data available. This may occur when there's insufficient data for the selected parameters.")
        
        

        # Create two columns for the tables
        table_col1, table_col2 = st.columns(2)
        
        with table_col1:
            st.markdown("**Daily Trade Details**")
            
            if hasattr(st.session_state, 'daily_df') and st.session_state.daily_df is not None:
                # Format the daily_df for better display
                daily_display = st.session_state.daily_df.copy()
                
                # Format columns for better readability
                if 'entry_date' in daily_display.columns:
                    daily_display['entry_date'] = pd.to_datetime(daily_display['entry_date']).dt.strftime('%Y-%m-%d')
                if 'expiry_date' in daily_display.columns:
                    daily_display['expiry_date'] = pd.to_datetime(daily_display['expiry_date']).dt.strftime('%Y-%m-%d')
                
                # Round numeric columns
                numeric_columns = ['entry_price', 'expiry_price', 'one_day_pct_move', 'expiry_pct_move', 
                                 'option_gains', 'option_price', 'underlying_pnl', 'current_pnl']
                for col in numeric_columns:
                    if col in daily_display.columns:
                        daily_display[col] = daily_display[col].round(2)
                
                # Format strike_prices column if it exists
                if 'strike_prices' in daily_display.columns:
                    daily_display['strike_prices'] = daily_display['strike_prices'].apply(
                        lambda x: f"{x[0]:.2f}" if isinstance(x, list) and len(x) > 0 else str(x)
                    )
                
                # Display with pagination
                st.dataframe(
                    daily_display,
                    use_container_width=True,
                    height=400,
                    hide_index=True
                )
                
                # Add download button for daily data
                csv_daily = daily_display.to_csv(index=False)
                st.download_button(
                    label="Download Daily Trade Data",
                    data=csv_daily,
                    file_name=f"{st.session_state.strategy_name}_{st.session_state.crypto_name}_daily_trades.csv",
                    mime="text/csv"
                )
            else:
                st.info("Daily trade data not available")
        
        with table_col2:
            st.markdown("**Weekday Performance Summary**")
            
            if hasattr(st.session_state, 'weekday_df') and st.session_state.weekday_df is not None:
                # Format the weekday_df for better display
                weekday_display = st.session_state.weekday_df.copy()
                
                # Round numeric columns
                numeric_columns = ['hits', 'losses', 'option_price', 'profit', 'total_profit', 'percentage_returns']
                for col in numeric_columns:
                    if col in weekday_display.columns:
                        weekday_display[col] = weekday_display[col].round(2)
                
                # Style the dataframe to highlight the average row
                def highlight_average_row(row):
                    if row['day_of_week'] == 'Average':
                        return ['background-color: #f0f2f6; font-weight: bold'] * len(row)
                    return [''] * len(row)
                
                styled_weekday = weekday_display.style.apply(highlight_average_row, axis=1)
                
                st.dataframe(
                    styled_weekday,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Add download button for weekday data
                csv_weekday = weekday_display.to_csv(index=False)
                st.download_button(
                    label="Download Weekday Summary",
                    data=csv_weekday,
                    file_name=f"{st.session_state.strategy_name}_{st.session_state.crypto_name}_weekday_summary.csv",
                    mime="text/csv"
                )
            else:
                st.info("Weekday summary data not available")
    
    else:
        st.info("Configure strategy parameters with sidebar and click 'Run Backtest' to see results.")
        
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

if __name__ == "__main__":
    main()