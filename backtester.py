import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from tabulate import tabulate
from scipy import stats
from math import erf

# Define options type to consistently check logic
class OptionType:
    CALL = "CALL"
    PUT = "PUT"

# Define position type to consistently check logic
class PositionType:
    LONG = "LONG"
    SHORT = "SHORT"

# Create option leg dataclass to easily add legs to create strategies - primarily used to calculate payoff for each leg
@dataclass
class OptionLeg:
    option_type: OptionType
    position_type: PositionType
    strike_pct: float  # Strike as % of spot (1.05 = 5% above spot)
    quantity: int = 1  # Number of contracts

    def calculate_strike(self, entry_price):
        """Calculate strike price based on entry price and strike percentage."""
        return entry_price * self.strike_pct

    def calculate_payoff(self, entry_price: float, expiry_price: float, premium: float):
        """Calculate payoff based on spot price at expiry."""
        strike = self.calculate_strike(entry_price)  # Calculate strike price based on given entry price and strike percentage

        # Calculate payoff based on option type and position type
        # CALL OPTION
        if self.option_type == OptionType.CALL:
            intrinsic = max(0.0, expiry_price - strike)
            if self.position_type == PositionType.LONG:  # Long call
                return intrinsic - premium  # Payoff if expiry price is above strike for long call
            else:  # Short call
                return premium - intrinsic  # Payoff if expiry price is below strike for short call

        # PUT OPTION
        else:
            intrinsic = max(0.0, strike - expiry_price)
            if self.position_type == PositionType.LONG:  # Long put
                return intrinsic - premium  # Payoff if expiry price is below strike for long put
            else:  # Short put
                return premium - intrinsic  # Payoff if expiry price is above strike for short put

    def calculate_pnl(self, entry_price: float, expiry_price: float, premium: float):
        """Calculate PnL based on quantity, entry price and priced premium."""
        payoff = self.calculate_payoff(entry_price, expiry_price, premium)
        return payoff * self.quantity

# Create options class container to hold multiple option legs to create strategies (i.e. covered call, straddle, etc.)
@dataclass
class OptionStrategy:
    """Container to hold defined multiple option legs to create strategies (e.g. covered call, straddle)."""
    name: str  # Name of the strategy (e.g. "Long Straddle", "Covered Call", "Iron Condor", etc.)
    legs: list  # List of options leg objects
    underlying_position: int = 0   # Number of shares to hold for strategy
    underlying_held: int = 0  # Number of underlying assets held  
    underlying_avg_cost: float = None  # Average cost of the underlying asset if held

    def calculate_total_pnl(self, entry_price: float, expiry_price: float, premiums_per_leg: list):
        """Calculate total PnL broken down by option legs and underlying."""
        # Option PnL across all legs
        option_pnl = 0.0
        strike_prices = []

        for leg, prem in zip(self.legs, premiums_per_leg or []):
            strike = leg.calculate_strike(entry_price)
            strike_prices.append(strike)
            option_pnl += leg.calculate_pnl(entry_price, expiry_price, prem)

        # Existing underlying PnL (if any)
        existing_underlying_pnl = 0.0
        if self.underlying_held and self.underlying_avg_cost is not None:
            existing_underlying_pnl = self.underlying_held * (expiry_price - self.underlying_avg_cost)

        # New underlying acquired for this strategy (e.g. covered call long 1 underlying)
        new_underlying_pnl = 0.0
        if self.underlying_position:
            new_underlying_pnl = self.underlying_position * (expiry_price - entry_price)

        # Total PnL for current trade
        trade_pnl = option_pnl + existing_underlying_pnl + new_underlying_pnl

        return {
            'trade_pnl': trade_pnl,
            'option_pnl': option_pnl,
            'existing_underlying_pnl': existing_underlying_pnl,
            'new_underlying_pnl': new_underlying_pnl,
            'strike_prices': strike_prices
        }

class OptionStrategyTemplate:
    """Collection of common option strategies with default parameters."""
    
    @staticmethod
    def covered_call(strike_pct: float = 1.05, underlying_position: int = 1, underlying_held: int = 0, underlying_avg_cost: float = None):
        """Create covered call strategy: Long underlying + Short 1 OTM Call. Premium is priced at entry via Black–Scholes."""
        return OptionStrategy(
            name="Covered Call",
            legs=[
                OptionLeg(
                    option_type=OptionType.CALL,
                    position_type=PositionType.SHORT,  # Sell call
                    strike_pct=strike_pct,
                    quantity=1
                )
            ],
            underlying_position=underlying_position,
            underlying_held=underlying_held,
            underlying_avg_cost=underlying_avg_cost
        )
    
    @staticmethod
    def cash_secured_put(strike_pct: float = 0.95):
        """Create cash secured put strategy: Short 1 OTM Put. Premium is priced at entry via Black–Scholes."""
        return OptionStrategy(
            name="Cash Secured Put",
            legs=[
                OptionLeg(
                    option_type=OptionType.PUT,
                    position_type=PositionType.SHORT,  # Sell put
                    strike_pct=strike_pct,
                    quantity=1
                )
            ],
            underlying_position=0  # No stock position
        )
    
    @staticmethod
    def long_straddle(strike_pct: float = 1.00):
        """Create long straddle: Long 1 ATM Call + Long 1 ATM Put. Premium is priced at entry via Black–Scholes."""
        return OptionStrategy(
            name="Long Straddle",
            legs=[
                OptionLeg(
                    option_type=OptionType.CALL,
                    position_type=PositionType.LONG,  # Buy call
                    strike_pct=strike_pct,
                    quantity=1
                ),
                OptionLeg(
                    option_type=OptionType.PUT,
                    position_type=PositionType.LONG,  # Buy put
                    strike_pct=strike_pct,
                    quantity=1
                )
            ],
            underlying_position=0  # No stock position
        )
    
    @staticmethod
    def iron_condor(put_short: float = 0.95, put_long: float = 0.90,
                    call_short: float = 1.05, call_long: float = 1.10):
        """Create iron condor: Short put spread + Short call spread. Premium is priced at entry via Black–Scholes."""
        return OptionStrategy(
            name="Iron Condor",
            legs=[
                # Put spread
                OptionLeg(  # Sell higher strike put
                    option_type=OptionType.PUT,
                    position_type=PositionType.SHORT,
                    strike_pct=put_short,
                    quantity=1
                ),
                OptionLeg(  # Buy lower strike put for protection
                    option_type=OptionType.PUT,
                    position_type=PositionType.LONG,
                    strike_pct=put_long,
                    quantity=1
                ),
                # Call spread
                OptionLeg(  # Sell lower strike call
                    option_type=OptionType.CALL,
                    position_type=PositionType.SHORT,
                    strike_pct=call_short,
                    quantity=1
                ),
                OptionLeg(  # Buy higher strike call for protection
                    option_type=OptionType.CALL,
                    position_type=PositionType.LONG,
                    strike_pct=call_long,
                    quantity=1
                )
            ],
            underlying_position=0  # No stock position
        )
    
    @staticmethod
    def bull_call_spread(long_strike: float = 1.00, short_strike: float = 1.05):
        """Create bull call spread: Long lower strike call + Short higher strike call. Premium is priced at entry via Black–Scholes."""
        return OptionStrategy(
            name="Bull Call Spread",
            legs=[
                OptionLeg(  # Buy lower strike call
                    option_type=OptionType.CALL,
                    position_type=PositionType.LONG,
                    strike_pct=long_strike,
                    quantity=1
                ),
                OptionLeg(  # Sell higher strike call
                    option_type=OptionType.CALL,
                    position_type=PositionType.SHORT,
                    strike_pct=short_strike,
                    quantity=1
                )
            ],
            underlying_position=0  # No stock position
        )
    
    @staticmethod
    def long_strangle(call_strike: float = 1.05, put_strike: float = 0.95):
        """Create long strangle: Long OTM Call + Long OTM Put. Premium is priced at entry via Black–Scholes."""
        return OptionStrategy(
            name="Long Strangle",
            legs=[
                OptionLeg(  # Buy OTM call
                    option_type=OptionType.CALL,
                    position_type=PositionType.LONG,
                    strike_pct=call_strike,
                    quantity=1
                ),
                OptionLeg(  # Buy OTM put
                    option_type=OptionType.PUT,
                    position_type=PositionType.LONG,
                    strike_pct=put_strike,
                    quantity=1
                )
            ],
            underlying_position=0  # No stock position
        )

# Backtester to run strategies against historical data
class OptionsBacktester:
    def __init__(self, price_data: pd.DataFrame):
        self.price_data = price_data.copy()

    # --- Black–Scholes helpers (per-unit premiums) ---
    @staticmethod
    def _N(x):
        """Standard normal CDF."""
        return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

    @staticmethod
    def _bs_call(S, K, r, sigma, T):
        """Black–Scholes call premium per unit."""
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return 0.0
        vol_sqrt_t = sigma * np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
        return S * OptionsBacktester._N(d1) - K * np.exp(-r * T) * OptionsBacktester._N(d2)

    @staticmethod
    def _bs_put(S, K, r, sigma, T):
        """Black–Scholes put premium per unit (put-call parity)."""
        call = OptionsBacktester._bs_call(S, K, r, sigma, T)
        return call - S + K * np.exp(-r * T)

    def _price_leg_premium(self, leg: OptionLeg, S: float, r: float, sigma: float, T: float):
        """Price a single option leg premium per unit at entry."""
        K = leg.calculate_strike(S)
        if leg.option_type == OptionType.CALL:
            return self._bs_call(S, K, r, sigma, T)
        else:
            return self._bs_put(S, K, r, sigma, T)

    def backtest_strategy(self, strategy: OptionStrategy, expiry_days: int = 7, 
                                start_date: str = None, end_date: str = None,
                                trade_frequency: str = 'non_overlapping',
                                interest_rate: float = 0.05, volatility: float = 0.50):
        """
        Full backtesting, analyses all dates for specificied time frame and provides detailed statistics.
        """
        data = self.price_data.copy()
        
        # Filter data by date range  
        data = data[data.index >= start_date] if start_date else data
        data = data[data.index <= end_date] if end_date else data
        
        daily_results = []
        
        # Process each day
        for i in range(len(data)):
            entry_date = data.index[i]
            entry_price = float(data.iloc[i]['close'])
            
            # Handle cases where we can't get next day or expiry data
            next_day_close = float(data.iloc[i + 1]['close']) if i + 1 < len(data) else entry_price
            expiry_price = float(data.iloc[i + expiry_days]['close']) if i + expiry_days < len(data) else entry_price
            expiry_date = data.index[i + expiry_days] if i + expiry_days < len(data) else entry_date

            one_day_pct_move = ((next_day_close - entry_price) / entry_price) * 100
            expiry_pct_move = ((expiry_price - entry_price) / entry_price) * 100
            
            # Calculate current strikes
            strike_prices = []
            for leg in strategy.legs:
                strike = leg.calculate_strike(entry_price)
                strike_prices.append(strike)
            
            # Calculate Black-Scholes option price for all days
            option_price = 0.0
            for leg in strategy.legs:
                strike = leg.calculate_strike(entry_price)
                T = expiry_days / 365.0  # Time to expiry in years
                
                # Get the premium using the Black-Scholes functions
                if leg.option_type == OptionType.CALL:
                    premium = self._bs_call(entry_price, strike, interest_rate, volatility, T)
                else:
                    premium = self._bs_put(entry_price, strike, interest_rate, volatility, T)
                
                # Apply position type (SHORT = negative, LONG = positive)
                if leg.position_type == PositionType.SHORT:
                    option_price += -premium * abs(leg.quantity)
                else:
                    option_price += premium * abs(leg.quantity)
                option_price = abs(option_price)
            
            # Initialise PnL
            option_hit = 0
            option_gains = 0.0
            old_strike = None
            
            # Check against strike from expiry_days ago
            if i >= expiry_days and len(daily_results) >= expiry_days:
                # Get the strike that was set expiry_days ago
                old_strike = daily_results[i - expiry_days+1]['strike_prices'][0]  # Assuming single leg for now
                
                # Check if today's close > old strike AND we have tomorrow's data
                if data.iloc[i]['close'] > old_strike and i + 1 < len(data):
                    option_hit = 1
                    # PnL = tomorrow's open - old strike
                    tomorrow_open = float(data.iloc[i + 1]['open'])
                    
                    # For SHORT position (like covered call) - LOSS
                    # For LONG position - GAIN
                    for leg in strategy.legs:
                        if leg.position_type == PositionType.SHORT:
                            # We lose money when the option is exercised against us
                            option_gains += -(tomorrow_open - old_strike) * abs(leg.quantity)
                        else:
                            # We make money when we exercise our long option
                            option_gains += (tomorrow_open - old_strike) * abs(leg.quantity)
                else:
                    option_hit = 0
                    option_gains = 0.0
            
            # Underlying PnL
            underlying_pnl = 0.0
            if strategy.underlying_position != 0:
                underlying_pnl = strategy.underlying_position * (expiry_price - entry_price)

            # Remove last 6 option prices - CHECK
            option_gains = np.nan if i >= len(data) - 6 else option_gains

            current_pnl = (option_price - abs(option_gains)) + underlying_pnl
            
            # Store results  
            daily_results.append({
                'entry_date': entry_date,
                'expiry_date': expiry_date,
                'entry_day_of_week': entry_date.strftime('%A'),
                'entry_price': entry_price,
                'expiry_price': expiry_price,
                'one_day_pct_move': round(one_day_pct_move, 2),
                'expiry_pct_move': round(expiry_pct_move, 2),
                'strike_prices': strike_prices,
                'old_strike': old_strike,
                'option_hit': option_hit,
                'option_gains': round(option_gains, 2),
                'option_price': round(option_price, 2),
                'underlying_pnl': round(underlying_pnl, 2),
                'current_pnl': round(current_pnl, 2),
            })
        
        daily_data = pd.DataFrame(daily_results)
        
        if daily_data.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Day of week performance summary
        weekday_stats = []
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for dow_name in weekdays:
            dow_data = daily_data[daily_data['entry_day_of_week'] == dow_name]

            hits = int(dow_data['option_hit'].sum())
            losses = abs(dow_data['option_gains'].sum()) 
            
            option_price = dow_data['option_price'].sum()

            profit = option_price - losses
            total_profit = profit + (data.iloc[-1]['close'] - data.iloc[0]['close'])

            percentage_returns = (total_profit / data.iloc[0]['close']) * 100
            
            stats = {
                'day_of_week': dow_name,
                'hits': hits,
                'losses': round(losses, 2),
                'option_price': round(option_price, 2),
                'profit': round(profit, 2), 
                'total_profit': round(total_profit, 2),
                'percentage_returns': round(percentage_returns, 2)
            }
            
            weekday_stats.append(stats)

        # Add average statistics
        if weekday_stats:
            avg_stats = {
                'day_of_week': 'Average',
                'hits': round(sum(stat['hits'] for stat in weekday_stats) / len(weekday_stats), 2),
                'losses': round(sum(stat['losses'] for stat in weekday_stats) / len(weekday_stats), 2),
                'option_price': round(sum(stat['option_price'] for stat in weekday_stats) / len(weekday_stats), 2),
                'profit': round(sum(stat['profit'] for stat in weekday_stats) / len(weekday_stats), 2),
                'total_profit': round(sum(stat['total_profit'] for stat in weekday_stats) / len(weekday_stats), 2),
                'percentage_returns': round(sum(stat['percentage_returns'] for stat in weekday_stats) / len(weekday_stats), 2)
            }
            weekday_stats.append(avg_stats)
        
        weekday_data = pd.DataFrame(weekday_stats)
        
        return daily_data, weekday_data

    def _calculate_max_drawdown(self, pnl_series):
        """Calculate maximum drawdown from cumulative PnL."""
        cumulative_pnl = pnl_series.cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        return drawdown.min()

    def _calculate_profit_factor(self, pnl_series):
        """Calculate profit factor (total profits / total losses)."""
        profits = pnl_series[pnl_series > 0].sum()
        losses = abs(pnl_series[pnl_series < 0].sum())
        return profits / losses if losses > 0 else float('inf')

    def _calculate_sharpe_ratio(self, returns, expiry_days):
        """Calculate annualised Sharpe ratio."""
        if returns.std() == 0:
            return 0
        mean_return = returns.mean()
        std_return = returns.std()
        # Annualise based on expiry days - Assuming risk-free rate = 0, ADD RISK-FREE RATE LATER) - CHECK IF WE NEED TO ANNUALISE
        periods_per_year = 252 / expiry_days  # Approximate trading periods per year
        return (mean_return / std_return) * np.sqrt(periods_per_year)

    def summary_stats(self, strategy: OptionStrategy, expiry_days: int = 7, start_date: str = None, end_date: str = None,
                    trade_frequency: str = 'non_overlapping', entry_day_of_week: int = None,
                    interest_rate: float = 0.05, volatility: float = 0.50):
        """Calculate summary statistics using the backtest_strategy logic with proper day filtering."""
        daily_data, weekday_data = self.backtest_strategy(strategy, expiry_days, start_date, end_date, 
                                                    trade_frequency, interest_rate, volatility)
        
        if daily_data.empty:
            return "No data available for the given date range."
        
        # Filter data based on entry_day_of_week if specified
        if entry_day_of_week is not None:
            # Convert day number to day name for filtering (1=Monday, 2=Tuesday, etc.)
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            if 1 <= entry_day_of_week <= 7:
                target_day = day_names[entry_day_of_week - 1]
                filter_data = daily_data[daily_data['entry_day_of_week'] == target_day].copy()
                if filter_data.empty:
                    return f"No data available for {target_day}s in the given date range."
            else:
                return "entry_day_of_week must be between 1 (Monday) and 7 (Sunday)."
        else:
            filter_data = daily_data.copy()
        
        # Filter out rows with NaN option_gains (last 6 days)
        valid_data = filter_data.dropna(subset=['option_gains']).copy()
        
        if valid_data.empty:
            return "No valid trades available for analysis."
        
        # Calculate cumulative PnL properly - as per user's logic
        valid_data['cumulative_pnl'] = valid_data['current_pnl'].cumsum()
        
        # Calculate total PnL for each day (option premium collected + option gains + underlying PnL)
        valid_data['total_pnl'] = valid_data['option_price'] + valid_data['option_gains'] + valid_data['underlying_pnl']
        
        # Calculate return percentage based on initial investment
        # For covered calls, initial investment includes underlying position
        valid_data['initial_investment'] = valid_data.apply(
            lambda row: abs(row['option_price']) + (abs(strategy.underlying_position) * row['entry_price'] if strategy.underlying_position != 0 else 0),
            axis=1
        )
        
        valid_data['return_pct'] = valid_data.apply(
            lambda row: (row['total_pnl'] / row['initial_investment']) * 100 if row['initial_investment'] != 0 else 0,
            axis=1
        )
        
        # Determine winning trades - options not exercised for short positions
        if len(strategy.legs) > 0 and strategy.legs[0].position_type == PositionType.SHORT:
            winning_trades = (valid_data['option_hit'] == 0).sum()  # Win when option not exercised
        else:
            winning_trades = (valid_data['total_pnl'] > 0).sum()  # Win when total PnL positive
        
        # Calculate summary statistics
        target_day_name = day_names[entry_day_of_week - 1] if entry_day_of_week else "All Days"
        
        stats = {
            'entry_date': f"{target_day_name} only" if entry_day_of_week else "All Days",
            'winning_days': winning_trades,
            'losing_days': len(valid_data) - winning_trades,
            'win_rate': (winning_trades / len(valid_data)) * 100,
            'total_profit': weekday_data[weekday_data['day_of_week'] == target_day_name]['total_profit'].values[0],
            'percentage_returns': weekday_data[weekday_data['day_of_week'] == target_day_name]['percentage_returns'].values[0],
            'best_day': valid_data['current_pnl'].max(),
            'worst_day': valid_data['current_pnl'].min(),
            'std_pnl': valid_data['current_pnl'].std(),
            'std_dev_returns': valid_data['return_pct'].std(),
            'max_drawdown': self._calculate_max_drawdown(valid_data['current_pnl']),
            'profit_factor': self._calculate_profit_factor(valid_data['current_pnl']),
            'sharpe_ratio': self._calculate_sharpe_ratio(valid_data['return_pct'], 1),  # Daily returns
        }
        
        # Round floats to 2 decimal places
        rounded_stats = {k: round(v, 2) if isinstance(v, (float, int)) and not np.isnan(v) else v for k, v in stats.items()}
        
        stats_table = tabulate(rounded_stats.items(), headers=["Metric", "Value"], tablefmt="pretty")
        return stats_table

    def plot_backtest_results(self, strategy: OptionStrategy, expiry_days: int = 7, start_date: str = None, end_date: str = None,
                            trade_frequency: str = 'non_overlapping', entry_day_of_week: int = None,
                            interest_rate: float = 0.05, volatility: float = 0.50):
        """Create visualisation of backtest results using backtest_strategy logic with proper day filtering."""
        daily_data, weekday_data = self.backtest_strategy(strategy, expiry_days, start_date, end_date, 
                                                    trade_frequency, interest_rate, volatility)
        
        # Filter data based on entry_day_of_week if specified
        if entry_day_of_week is not None:
            # Convert day number to day name for filtering (1=Monday, 2=Tuesday, etc.)
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            if 1 <= entry_day_of_week <= 7:
                target_day = day_names[entry_day_of_week - 1]
                filter_data = daily_data[daily_data['entry_day_of_week'] == target_day].copy()
                if filter_data.empty:
                    print(f"No data available for {target_day}s in the given date range.")
                    return None
            else:
                print("entry_day_of_week must be between 1 (Monday) and 7 (Sunday).")
                return None
        else:
            filter_data = daily_data.copy()
        
        # Filter out rows with NaN option_gains
        results = filter_data.dropna(subset=['option_gains']).copy()
        
        if results.empty:
            print("No valid data to plot")
            return None
        
        # Calculate cumulative PnL
        results['cumulative_pnl'] = results['current_pnl'].cumsum()

        # Calculate total PnL and return metrics
        results['total_pnl'] = results['option_price'] + results['option_gains'] + results['underlying_pnl']
        results['initial_investment'] = results.apply(
            lambda row: abs(row['option_price']) + (abs(strategy.underlying_position) * row['entry_price'] if strategy.underlying_position != 0 else 0),
            axis=1
        )
        results['return_pct'] = results.apply(
            lambda row: (row['total_pnl'] / row['initial_investment']) * 100 if row['initial_investment'] != 0 else 0,
            axis=1
        )
        results['log_return'] = results.apply(
            lambda row: np.log(1 + (row['total_pnl'] / row['initial_investment'])) * 100
            if row['initial_investment'] != 0 and (1 + row['total_pnl'] / row['initial_investment']) > 0
            else np.nan, axis=1
        )
        
        target_day_name = day_names[entry_day_of_week - 1] if entry_day_of_week else "All Days"
        
        # Determine underlying asset name from price data columns or use generic term
        underlying_name = "BTC"  # Default to BTC, but this could be made configurable
        if hasattr(self.price_data, 'columns') and len(self.price_data.columns) > 0:
            # Try to infer from column names or use a more generic approach
            underlying_name = "Underlying Asset"
        
        # Create subplots with new layout
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                f'{underlying_name} Price', 'Cumulative PnL',
                'Daily PnL Distribution', 'Rolling Performance Metrics',
                'Entry Price vs Exercise Events', 'Monthly PnL'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"type": "histogram"}, {"secondary_y": True}],
                [{"type": "scatter"}, {"type": "bar"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # 1. Underlying Price (Top Left) - use original price data
        price_data_filtered = self.price_data.copy()
        if start_date:
            price_data_filtered = price_data_filtered[price_data_filtered.index >= start_date]
        if end_date:
            price_data_filtered = price_data_filtered[price_data_filtered.index <= end_date]
        
        fig.add_trace(
            go.Scatter(x=price_data_filtered.index, y=price_data_filtered['close'],
                    mode='lines', name=f'{underlying_name} Price',
                    line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # 2. Cumulative PnL (Top Right) with Buy & Hold comparison
        fig.add_trace(
            go.Scatter(x=results['entry_date'], y=results['cumulative_pnl'],
                    mode='lines', name='Options Strategy',
                    line=dict(color='green', width=3)),
            row=1, col=2
        )
        
        # Calculate buy and hold returns
        if not results.empty:
            initial_price = results['entry_price'].iloc[0]
            buy_hold_cumulative = []
            for _, row in results.iterrows():
                # Buy and hold assuming 1 share return = (current_price - initial_price) * number of shares
                buy_hold_return = (row['entry_price'] - initial_price)
                buy_hold_cumulative.append(buy_hold_return)
            
            fig.add_trace(
                go.Scatter(x=results['entry_date'], y=buy_hold_cumulative,
                        mode='lines', name='Buy & Hold',
                        line=dict(color='blue', width=2)),
                row=1, col=2
            )
        
        # Add zero line for reference
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
        
        # 3. Daily PnL Distribution (Middle Left) with proper color coding
        # Use plotly's built-in histogram with custom colors based on bin positions
        pnl_values = results['current_pnl'].values
        
        # Create histogram data manually to control colors properly
        counts, bin_edges = np.histogram(pnl_values, bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Create separate traces for positive and negative bins
        for i, (count, left_edge, right_edge, center) in enumerate(zip(counts, bin_edges[:-1], bin_edges[1:], bin_centers)):
            if count > 0:  # Only plot if there's data in this bin
                color = 'green' if center >= 0 else 'red'
                
                fig.add_trace(
                    go.Bar(x=[center], y=[count],
                        width=[right_edge - left_edge],
                        marker_color=color,
                        opacity=0.7,
                        showlegend=False),
                    row=2, col=1
                )
        
        # 4. Rolling Performance Metrics (Middle Right)
        results_indexed = results.set_index('entry_date').sort_index()
        rolling_pnl = results_indexed['current_pnl'].rolling('30D').mean()
        rolling_vol = results_indexed['return_pct'].rolling('30D').std()
        
        rolling_pnl = rolling_pnl.dropna()
        rolling_vol = rolling_vol.dropna()
        
        if not rolling_pnl.empty:
            fig.add_trace(
                go.Scatter(x=rolling_pnl.index, y=rolling_pnl.values,
                        mode='lines', name='30D Avg PnL',
                        line=dict(color='blue')),
                row=2, col=2
            )
        
        if not rolling_vol.empty:
            fig.add_trace(
                go.Scatter(x=rolling_vol.index, y=rolling_vol.values,
                        mode='lines', name='30D Volatility',
                        line=dict(color='red')),
                row=2, col=2, secondary_y=True
            )
        
        # 5. Entry Price vs Exercise Events (Bottom Left)
        colors = ['red' if hit else 'green' for hit in results['option_hit']]
        fig.add_trace(
            go.Scatter(x=results['entry_price'], y=results['expiry_price'],
                    mode='markers', name='Exercise Events',
                    marker=dict(color=colors, size=5),
                    text=[f"Exercised: {'Yes' if hit else 'No'}" for hit in results['option_hit']],
                    hovertemplate="Entry: $%{x}<br>Expiry: $%{y}<br>%{text}<extra></extra>"),
            row=3, col=1
        )
        
        # Add diagonal line for reference
        min_price = min(results['entry_price'].min(), results['expiry_price'].min())
        max_price = max(results['entry_price'].max(), results['expiry_price'].max())
        fig.add_trace(
            go.Scatter(x=[min_price, max_price], y=[min_price, max_price],
                    mode='lines', line=dict(dash='dash', color='gray'),
                    showlegend=False),
            row=3, col=1
        )
        
        # 6. Monthly PnL Bar Chart (Bottom Right)
        results['year'] = results['entry_date'].dt.year
        results['month_num'] = results['entry_date'].dt.month
        results['year_month'] = results['entry_date'].dt.to_period('M')
        
        monthly_pnl = results.groupby('year_month')['current_pnl'].sum().reset_index()
        monthly_pnl['year_month_str'] = monthly_pnl['year_month'].astype(str)
        
        # Color bars based on positive/negative values
        bar_colors = ['green' if pnl >= 0 else 'red' for pnl in monthly_pnl['current_pnl']]
        
        fig.add_trace(
            go.Bar(x=monthly_pnl['year_month_str'], y=monthly_pnl['current_pnl'],
                name='Monthly PnL',
                marker_color=bar_colors,
                text=[f"${pnl:.0f}" for pnl in monthly_pnl['current_pnl']],
                textposition='outside'),
            row=3, col=2
        )
        
        # Add zero line for monthly PnL
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=3, col=2)
        
        fig.update_layout(
            title=f'{strategy.name} - Backtest Analysis ({target_day_name})',
            showlegend=False,
            height=1200,
            width=1400,
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text=f"{underlying_name} Price ($)", row=1, col=1)
        
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative PnL ($)", row=1, col=2)
        
        fig.update_xaxes(title_text="Daily PnL ($)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="PnL ($)", row=2, col=2)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=2, secondary_y=True)
        
        fig.update_xaxes(title_text="Entry Price ($)", row=3, col=1)
        fig.update_yaxes(title_text="Expiry Price ($)", row=3, col=1)
        
        fig.update_xaxes(title_text="Month", row=3, col=2)
        fig.update_yaxes(title_text="Monthly PnL ($)", row=3, col=2)
        
        return fig