import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from tabulate import tabulate
from scipy import stats

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
    premium_pct: float  # Premium as % of spot
    quantity: int = 1  # Number of contracts

    def calculate_strike(self, entry_price):
        """Calculate strike price based on entry price and strike percentage."""
        return entry_price * self.strike_pct
    
    def calculate_premium(self, entry_price):
        """Calculate premium based on entry price and premium percentage."""
        return entry_price * self.premium_pct

    def calculate_payoff(self, entry_price: float, expiry_price: float):
        """Calculate payoff based on spot price at expiry."""
        strike = self.calculate_strike(entry_price)  # Calculate strike price based on given entry price and strike percentage
        premium = self.calculate_premium(entry_price)  # Calculate premium based on given entry price and premium percentage

        # Calculate payoff based on option type and position type
        # CALL OPTION
        if self.option_type == OptionType.CALL:
            if self.position_type == PositionType.LONG:  # Long call
                return max(0, expiry_price - strike) - premium # Payoff if expiry price is above strike for long call
            else:  # Short call
                return premium - max(0, expiry_price - strike)  # Payoff if expiry price is below strike for short call

        # PUT OPTION
        else:  
            if self.position_type == PositionType.LONG:  # Long put
                return max(0, strike - expiry_price) - premium  # Payoff if expiry price is below strike for long put
            else:  # Short put
                return premium - max(0, strike - expiry_price)  # Payoff if expiry price is above strike for short put

    def calculate_pnl(self, entry_price: float, expiry_price: float):
        """Calculate PnL based on quantity, entry price and expiry price."""
        payoff = self.calculate_payoff(entry_price, expiry_price)
        return payoff * self.quantity

# Create options class container to hold multiple option legs to create strategies (i.e. covered call, straddle, etc.)
class OptionStrategy:
    """Container to hold defined multiple option legs to create strategies (e.g. covered call, straddle)."""
    def __init__(self, name: str, legs: list[OptionLeg], underlying_position: int = 0, underlying_held: int = 0, underlying_avg_cost: float = 0.0):
        self.name = name  # Name of the strategy (e.g. "Long Straddle", "Covered Call", "Iron Condor", etc.)
        self.legs = legs  # List of options leg objects
        self.underlying_position = underlying_position  # Number of shares to hold for strategy
        self.underlying_held = underlying_held  # Number of underlying assets held  
        self.underlying_avg_cost = underlying_avg_cost  # Average cost of the underlying asset if held

        # Calculate how many new shares we need to buy/sell
        self.underlying_position_buy = underlying_position - underlying_held

    def calculate_total_pnl(self, entry_price: float, expiry_price: float):
        """Calculate total PnL for the strategy based on entry and expiry prices."""
        # PnL from existing shares (held before strategy) if provided
        underlying_held_pnl = 0.0
        if self.underlying_held != 0:
            if self.underlying_avg_cost is not None:
                underlying_held_pnl = self.underlying_held * (expiry_price - self.underlying_avg_cost)
        # PnL from new shares
        underlying_position_pnl = 0.0
        if self.underlying_position_buy != 0:  # Can be positive (buy) or negative (sell)
            underlying_position_pnl = self.underlying_position_buy * (expiry_price - entry_price)
        # Calculate total underlying PnL if holding underlying asset
        underlying_pnl = 0.0
        if self.underlying_position > 0:
            underlying_pnl = underlying_held_pnl + underlying_position_pnl

        # Calculate PnL from all option legs
        option_pnl = sum(leg.calculate_pnl(entry_price, expiry_price) for leg in self.legs)

        # Calculate total PnL
        trade_pnl = underlying_pnl + option_pnl

        # Calculate underlying asset cost if underlying purchased
        underlying_cost = self.underlying_position * entry_price
        # Calculate total premium paid for options
        total_premium = sum(leg.calculate_premium(entry_price) * leg.quantity for leg in self.legs)
        # Calculate initial investment (underlying position cost + total premium paid for options)
        initial_investment = underlying_cost + total_premium

        return {
            'underlying_held_pnl': underlying_held_pnl,
            'underlying_position_pnl': underlying_position_pnl,
            'underlying_pnl': underlying_pnl,
            'options_pnl': option_pnl,
            'trade_pnl': trade_pnl,
            'initial_investment': initial_investment
        }

# Create option strategy templates to easily create common strategies
class OptionStrategyTemplate:
    """Collection of common option strategies with default parameters."""
    
    @staticmethod
    def covered_call(strike_pct: float = 1.05, premium_pct: float = 0.02, underlying_position: int=100, underlying_held: int = 0, underlying_avg_cost: float = None):
        """Create covered call strategy: Long 100 shares + Short 1 OTM Call."""
        return OptionStrategy(
            name="Covered Call",
            legs=[
                OptionLeg(
                    option_type=OptionType.CALL,
                    position_type=PositionType.SHORT,  # Sell call
                    strike_pct=strike_pct,  # Default 5% OTM
                    premium_pct=premium_pct,  # Premium collected
                    quantity=1
                )
            ],
            underlying_position=underlying_position,
            underlying_held=underlying_held,
            underlying_avg_cost=underlying_avg_cost
        )
    
    @staticmethod
    def cash_secured_put(strike_pct: float = 0.95, premium_pct: float = 0.02):
        """Create cash secured put strategy: Short 1 OTM Put."""
        return OptionStrategy(
            name="Cash Secured Put",
            legs=[
                OptionLeg(
                    option_type=OptionType.PUT,
                    position_type=PositionType.SHORT,  # Sell put
                    strike_pct=strike_pct,  # Default 5% OTM
                    premium_pct=premium_pct,  # Premium collected
                    quantity=1
                )
            ],
            underlying_position=0  # No stock position
        )
    
    @staticmethod
    def long_straddle(strike_pct: float = 1.00, call_premium: float = 0.03, put_premium: float = 0.03):
        """Create long straddle: Long 1 ATM Call + Long 1 ATM Put."""
        return OptionStrategy(
            name="Long Straddle",
            legs=[
                OptionLeg(
                    option_type=OptionType.CALL,
                    position_type=PositionType.LONG,  # Buy call
                    strike_pct=strike_pct,  # ATM strike
                    premium_pct=call_premium,  # Premium paid
                    quantity=1
                ),
                OptionLeg(
                    option_type=OptionType.PUT,
                    position_type=PositionType.LONG,  # Buy put
                    strike_pct=strike_pct,  # Same strike as call
                    premium_pct=put_premium,  # Premium paid
                    quantity=1
                )
            ],
            underlying_position=0  # No stock position
        )
    
    @staticmethod
    def iron_condor(put_short: float = 0.95, put_long: float = 0.90,
                    call_short: float = 1.05, call_long: float = 1.10,
                    short_premium: float = 0.02, long_premium: float = 0.01):
        """Create iron condor: Short put spread + Short call spread."""
        return OptionStrategy(
            name="Iron Condor",
            legs=[
                # Put spread
                OptionLeg(  # Sell higher strike put
                    option_type=OptionType.PUT,
                    position_type=PositionType.SHORT,
                    strike_pct=put_short,  # 5% OTM put
                    premium_pct=short_premium,  # Premium collected
                    quantity=1
                ),
                OptionLeg(  # Buy lower strike put for protection
                    option_type=OptionType.PUT,
                    position_type=PositionType.LONG,
                    strike_pct=put_long,  # 10% OTM put
                    premium_pct=long_premium,  # Premium paid
                    quantity=1
                ),
                # Call spread
                OptionLeg(  # Sell lower strike call
                    option_type=OptionType.CALL,
                    position_type=PositionType.SHORT,
                    strike_pct=call_short,  # 5% OTM call
                    premium_pct=short_premium,  # Premium collected
                    quantity=1
                ),
                OptionLeg(  # Buy higher strike call for protection
                    option_type=OptionType.CALL,
                    position_type=PositionType.LONG,
                    strike_pct=call_long,  # 10% OTM call
                    premium_pct=long_premium,  # Premium paid
                    quantity=1
                )
            ],
            underlying_position=0  # No stock position
        )
    
    @staticmethod
    def bull_call_spread(long_strike: float = 1.00, short_strike: float = 1.05,
                        long_premium: float = 0.03, short_premium: float = 0.01):
        """Create bull call spread: Long lower strike call + Short higher strike call."""
        return OptionStrategy(
            name="Bull Call Spread",
            legs=[
                OptionLeg(  # Buy lower strike call
                    option_type=OptionType.CALL,
                    position_type=PositionType.LONG,
                    strike_pct=long_strike,  # ATM or slightly ITM
                    premium_pct=long_premium,  # Premium paid
                    quantity=1
                ),
                OptionLeg(  # Sell higher strike call
                    option_type=OptionType.CALL,
                    position_type=PositionType.SHORT,
                    strike_pct=short_strike,  # OTM
                    premium_pct=short_premium,  # Premium collected
                    quantity=1
                )
            ],
            underlying_position=0  # No stock position
        )
    
    @staticmethod
    def long_strangle(call_strike: float = 1.05, put_strike: float = 0.95,
                     call_premium: float = 0.015, put_premium: float = 0.015):
        """Create long strangle: Long OTM Call + Long OTM Put."""
        return OptionStrategy(
            name="Long Strangle",
            legs=[
                OptionLeg(  # Buy OTM call
                    option_type=OptionType.CALL,
                    position_type=PositionType.LONG,
                    strike_pct=call_strike,  # OTM call
                    premium_pct=call_premium,  # Premium paid
                    quantity=1
                ),
                OptionLeg(  # Buy OTM put
                    option_type=OptionType.PUT,
                    position_type=PositionType.LONG,
                    strike_pct=put_strike,  # OTM put
                    premium_pct=put_premium,  # Premium paid
                    quantity=1
                )
            ],
            underlying_position=0  # No stock position
        )

# Backtester to run strategies against historical data
class OptionsBacktester:
    def __init__(self, price_data: pd.DataFrame):
        self.price_data = price_data.copy()

    def backtest_strategy(self, strategy: OptionStrategy, expiry_days: int = None, start_date: str = None, end_date: str = None, trade_frequency: str = 'non_overlapping', entry_day_of_week: int = None):
        '''Systematic backtesting of option strategy over given date range - loops and places trades on each date (CHANGE SO RULE BASED)'''
        data = self.price_data.copy()
        # Filter data by date range provided
        data = data[data.index >= start_date] if start_date else data
        data = data[data.index <= end_date] if end_date else data

        trade_result = []  # Store results for each trade
        i = 0
        
        # Loop through data with proper trade frequency logic
        while i < len(data) - expiry_days:
            # Get entry date for current trade
            entry_date = data.index[i]
            
            # Check if we should enter a trade based on frequency rules - i.e. place trade daily or weekly or monthly
            trade_cond = False
            if trade_frequency == 'daily':
                trade_cond = True
            elif trade_frequency == 'weekly':
                trade_cond = entry_date.dayofweek == (entry_day_of_week or 0)  # Default Monday = 0
            elif trade_frequency == 'monthly':
                trade_cond = entry_date.day <= 7 and entry_date.dayofweek == (entry_day_of_week or 0)  # First week of month
            elif trade_frequency == 'non_overlapping':
                trade_cond = True  # Jump by expiry_days after each trade - no overlap
            else:
                raise ValueError("Invalid trade frequency. Choose from 'daily', 'weekly', 'monthly', or 'non_overlapping'. CONSIDER ADDING MORE.")
            
            if trade_cond:
                # Get entry details
                entry_price = float(data.iloc[i]['close'])

                # Get expiry details
                expiry_date = data.index[i + expiry_days]
                expiry_price = float(data.iloc[i + expiry_days]['close'])

                # Calculate current PnL for this trade
                pnl_breakdown = strategy.calculate_total_pnl(entry_price, expiry_price)

                # Calculate metrics
                price_change = expiry_price - entry_price  # Absolute price change
                price_change_pct = (price_change / entry_price) * 100  # Percentage price change
                
                # Track which option legs expired ITM - generalised logic for option calculations - CHECK THIS LATER
                legs_expired_itm = []
                strike_prices = []
                for leg in strategy.legs:
                    strike = leg.calculate_strike(entry_price)
                    strike_prices.append(strike)

                    # Check if option expired ITM based on option type
                    if leg.option_type == OptionType.CALL:
                        expired_itm = expiry_price > strike  # Call ITM when current price above strike
                    else:  # OptionType.PUT
                        expired_itm = expiry_price < strike  # Put ITM when current price below strike
                
                    legs_expired_itm.append({
                        'option_type': leg.option_type,
                        'position_type': leg.position_type,
                        'strike_price': strike,
                        'expired_itm': expired_itm,
                    })

                # Determine if primary option leg expired ITM (simple win/loss tracking)
                primary_leg_expired_itm = legs_expired_itm[0]['expired_itm'] if legs_expired_itm else False

                # Store results for current trade
                trade_result.append({
                    'entry_date': entry_date,
                    'expiry_date': expiry_date,
                    'entry_price': entry_price,
                    'expiry_price': expiry_price,
                    'strike_price': strike_prices,
                    'legs_expired_itm': legs_expired_itm,
                    'primary_leg_expired_itm': primary_leg_expired_itm,
                    'price_change': price_change,
                    'price_change_pct': price_change_pct,
                    'underlying_position': strategy.underlying_position,
                    'underlying_cost': strategy.underlying_position * entry_price if strategy.underlying_position > 0 else 0,
                    'underlying_held': strategy.underlying_held,
                    'underlying_avg_cost': strategy.underlying_avg_cost,
                    'underlying_held_pnl': pnl_breakdown['underlying_held_pnl'],
                    'underlying_position_pnl': pnl_breakdown['underlying_position_pnl'],
                    'underlying_pnl': pnl_breakdown['underlying_pnl'],
                    'options_pnl': pnl_breakdown['options_pnl'],
                    'trade_pnl': pnl_breakdown['trade_pnl'],
                    'initial_investment': pnl_breakdown['initial_investment'],
                })
                
                # Move to next trade date
                if trade_frequency == 'non_overlapping':
                    i += expiry_days  # Jump to after expiry
                else:
                    i += 1  # Move to next day
            else:
                i += 1  # Move to next day if not trading

        return pd.DataFrame(trade_result)

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

    def summary_stats(self, strategy: OptionStrategy, expiry_days: int = None, start_date: str = None, end_date: str = None, trade_frequency: str = 'non_overlapping', entry_day_of_week: int = None):
        """Calculate summary statistics for the backtested strategy."""
        trades = self.backtest_strategy(strategy, expiry_days, start_date, end_date, trade_frequency, entry_day_of_week)
        
        if trades.empty:
            return "No trades executed in the given date range."
        
        # Calculate return percentage for each trade
        trades = trades.copy()
        trades['return_pct'] = trades.apply(lambda row: (row['trade_pnl'] / abs(row['initial_investment'])) * 100 
                                                if row['initial_investment'] != 0 else 0, axis=1)

        # Determine winning trades based on strategy type - generalised for option types and strategies CHECK THIS LATER
        winning_trades = 0
        if len(strategy.legs) > 0:
            primarily_leg = strategy.legs[0]  # Assume first leg is primary for win/loss tracking

            # Option sellers - win when option expires OTM
            if primarily_leg.position_type == PositionType.SHORT:
                winning_trades = (~trades['primary_leg_expired_itm']).sum()
            # Option buyers - win when option expires ITM
            else:
                winning_trades = (trades['trade_pnl'] > 0).sum()
        else:
            # No option legs - treat all trades as winning if P&L is positive
            winning_trades = (trades['trade_pnl'] > 0).sum()

        # Calculate summary statistics
        stats = {
            'total_trades': len(trades),
            'winning_trades': winning_trades,
            'losing_trades': len(trades) - winning_trades,
            'win_rate': (winning_trades / len(trades)) * 100,  # As percentage
            'avg_pnl': trades['trade_pnl'].mean(),
            'total_pnl': trades['trade_pnl'].sum(),
            'avg_return': trades['return_pct'].mean(),  # Average return percentage
            'total_return': trades['return_pct'].sum(),  # Total return percentage
            'best_trade': trades['trade_pnl'].max(),
            'worst_trade': trades['trade_pnl'].min(),
            'best_trade_return': trades['return_pct'].max(),
            'worst_trade_return': trades['return_pct'].min(),
            'std_pnl': trades['trade_pnl'].std(),
            'std_return': trades['return_pct'].std(),
            'max_drawdown': self._calculate_max_drawdown(trades['trade_pnl']),
            'profit_factor': self._calculate_profit_factor(trades['trade_pnl']),
            'sharpe_ratio': self._calculate_sharpe_ratio(trades['return_pct'], expiry_days),
            'options_expired_itm': trades['primary_leg_expired_itm'].sum() if 'primary_leg_expired_itm' in trades else 0  # Track ITM expiries
        }

        # Round floats to 2 decimal places
        rounded_stats = {k: round(v, 2) if isinstance(v, (float, int)) else v for k, v in stats.items()}

        stats_table = tabulate(rounded_stats.items(), headers=["Metric", "Value"], tablefmt="pretty")
        return stats_table

    def plot_backtest_results(self, strategy: OptionStrategy, expiry_days: int = None, start_date: str = None, end_date: str = None, trade_frequency: str = 'non_overlapping', entry_day_of_week: int = None):
        """Create visualisation of backtest results."""
        # Run backtest to get results
        results = self.backtest_strategy(strategy, expiry_days, start_date, end_date, trade_frequency, entry_day_of_week)
        
        if results.empty:
            print("No trades to plot")
            return None
        
        # Calculate return percentage and log returns for plotting
        results = results.copy()
        results['return_pct'] = results.apply(
            lambda row: (row['trade_pnl'] / abs(row['initial_investment'])) * 100 
            if row['initial_investment'] != 0 else 0, axis=1
        )
        
        # Calculate log returns - better for time series analysis
        results['log_return'] = results.apply(
            lambda row: np.log(1 + (row['trade_pnl'] / abs(row['initial_investment']))) * 100
            if row['initial_investment'] != 0 and (1 + row['trade_pnl'] / abs(row['initial_investment'])) > 0 
            else np.nan, axis=1
        )
        
        # Create subplots with 4x2 grid
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Cumulative P&L Over Time', 'P&L & Log Return Distribution',
                'Win Rate by Month', 'Average Log Return by Day of Week',
                'Price Movement vs P&L', 'Rolling 30-Day Log Performance',
                'Entry vs Expiry Prices', 'P&L Heatmap by Month'
            ),
            specs=[
                [{"secondary_y": False}, {"type": "histogram"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"secondary_y": True}],
                [{"type": "scatter"}, {"type": "heatmap"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # 1. Enhanced Cumulative P&L over time with benchmark
        cumulative_pnl = results['trade_pnl'].cumsum()

        # Calculate strategy-matched benchmark (buy underlying at same frequency)
        results['underlying_return'] = ((results['expiry_price'] - results['entry_price']) / results['entry_price']) * abs(results['initial_investment'])
        cumulative_benchmark = results['underlying_return'].cumsum()
        
        # Calculate cumulative log returns for additional insight
        results_sorted = results.sort_values('entry_date')
        cumulative_log_returns = results_sorted['log_return'].fillna(0).cumsum()

        # Plot strategy performance
        fig.add_trace(
            go.Scatter(x=results['entry_date'], y=cumulative_pnl, 
                    mode='lines', name='Strategy P&L',
                    line=dict(color='green', width=2)),
            row=1, col=1
        )

        # Plot benchmark performance - buy & hold at same frequency
        fig.add_trace(
            go.Scatter(x=results['entry_date'], y=cumulative_benchmark, 
                    mode='lines', name='Buy & Hold (Same Frequency)',
                    line=dict(color='blue', width=2, dash='dot')),
            row=1, col=1
        )

        # Add a zero line for reference
        fig.add_hline(y=0, line_color="gray", 
                    opacity=0.5, row=1, col=1)

        # Add outperformance shading
        outperformance = cumulative_pnl - cumulative_benchmark
        fig.add_trace(
            go.Scatter(x=results['entry_date'], y=outperformance,
                    mode='lines', name='Outperformance',
                    line=dict(color='purple', width=1),
                    fill='tonexty', fillcolor='rgba(128,0,128,0.1)'),
            row=1, col=1
        )
        
        # Add log return distribution
        fig.add_trace(
            go.Histogram(x=results['log_return'].dropna(), nbinsx=30, 
                        name='Log Return Distribution (%)',
                        marker_color='lightcoral',
                        opacity=0.7),
            row=1, col=2
        )
        
        # 3. Win rate by month based on strategy type - generalised logic
        results['month'] = results['entry_date'].dt.to_period('M')
        if len(strategy.legs) > 0 and strategy.legs[0].position_type == PositionType.SHORT:
            # For option sellers - win = option expired OTM
            monthly_stats = results.groupby('month').agg({
                'primary_leg_expired_itm': lambda x: (~x).mean() * 100  # Win rate = expired OTM
            }).reset_index()
            monthly_stats.rename(columns={'primary_leg_expired_itm': 'win_rate'}, inplace=True)
        else:
            # For option buyers and other strategies - win = positive P&L
            monthly_stats = results.groupby('month').agg({
                'trade_pnl': lambda x: (x > 0).mean() * 100  # Win rate = positive P&L
            }).reset_index()
            monthly_stats.rename(columns={'trade_pnl': 'win_rate'}, inplace=True)
        
        monthly_stats['month_str'] = monthly_stats['month'].astype(str)
        
        fig.add_trace(
            go.Bar(x=monthly_stats['month_str'], y=monthly_stats['win_rate'],
                name='Win Rate %', marker_color='purple'),
            row=2, col=1
        )
        
        # 4. Average log return by day of week
        results['dow'] = results['entry_date'].dt.day_name()
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_log_stats = results.groupby('dow')['log_return'].mean().reindex(dow_order)
        
        fig.add_trace(
            go.Bar(x=dow_log_stats.index, y=dow_log_stats.values,
                name='Avg Log Return %', marker_color='orange'),
            row=2, col=2
        )
        
        # 5. Price movement vs P&L scatter plot
        fig.add_trace(
            go.Scatter(x=results['price_change_pct'], y=results['trade_pnl'],
                    mode='markers', name='P&L vs Price Move',
                    marker=dict(color=results['trade_pnl'], colorscale='RdYlGn', 
                                showscale=False, size=5)),  # CHANGE SHOWSCALE FOR COLOURBAR
            row=3, col=1
        )
        
        # 6. Rolling 30-day performance metrics
        # Ensure entry_date is datetime and set as index for rolling calculations
        results_copy = results.set_index('entry_date').sort_index()

        # Use log returns for rolling metrics
        rolling_log_return = results_copy['log_return'].rolling('30D').mean()
        rolling_vol = results_copy['return_pct'].rolling('30D').std()

        # Drop NaN values for cleaner plotting
        rolling_log_return = rolling_log_return.dropna()
        rolling_vol = rolling_vol.dropna()

        fig.add_trace(
            go.Scatter(x=rolling_log_return.index, y=rolling_log_return.values,
                    mode='lines', name='30D Avg Log Return (%)',
                    line=dict(color='blue')),
            row=3, col=2
        )

        fig.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol.values,
                    mode='lines', name='30D Volatility (%)',
                    line=dict(color='red')),
            row=3, col=2, secondary_y=True
        )
        
        # 7. Entry vs expiry prices scatter with colour coding based on ITM status
        if 'primary_leg_expired_itm' in results and len(strategy.legs) > 0:
            if strategy.legs[0].position_type == PositionType.SHORT:
                # For sellers - red if ITM (bad), green if OTM (good)
                colors = ['red' if itm else 'green' for itm in results['primary_leg_expired_itm']]
            else:
                # For buyers - green if profitable, red if not
                colors = ['green' if pnl > 0 else 'red' for pnl in results['trade_pnl']]
        else:
            # Default to P&L based colouring
            colors = ['green' if pnl > 0 else 'red' for pnl in results['trade_pnl']]
        
        fig.add_trace(
            go.Scatter(x=results['entry_price'], y=results['expiry_price'],
                    mode='markers', name='Entry vs Expiry',
                    marker=dict(color=colors, size=5)),
            row=4, col=1
        )
        
        # Add diagonal reference line
        min_price = min(results['entry_price'].min(), results['expiry_price'].min())
        max_price = max(results['entry_price'].max(), results['expiry_price'].max())
        fig.add_trace(
            go.Scatter(x=[min_price, max_price], y=[min_price, max_price],
                    mode='lines', line=dict(dash='dash', color='gray'),
                    showlegend=False),
            row=4, col=1
        )
        
        # Add strike lines if available
        if 'strike_prices' in results and len(strategy.legs) > 0:
            for i, leg in enumerate(strategy.legs):
                if i == 0:  # Only plot primary leg strike
                    strike_prices = [strikes[i] if i < len(strikes) else None 
                                for strikes in results['strike_prices']]
                    fig.add_trace(
                        go.Scatter(x=results['entry_price'], y=strike_prices,
                                mode='markers', name=f'{leg.option_type} Strike',
                                marker=dict(color='blue', size=3, symbol='x')),
                        row=4, col=1
                    )
        
        # 8. Improved P&L heatmap by month and year
        results['year'] = results['entry_date'].dt.year
        results['month_num'] = results['entry_date'].dt.month

        # Create pivot table and handle NaN values
        heatmap_data = results.pivot_table(
            values='trade_pnl', 
            index='month_num', 
            columns='year', 
            aggfunc='sum'
        ).fillna(0)  # Replace NaN with 0

        # Create month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # Ensure we have all 12 months in the index
        full_month_index = list(range(1, 13))
        heatmap_data = heatmap_data.reindex(full_month_index, fill_value=0)

        # Create text annotations - show empty string for zero values (so it doesn't show NaN)
        text_annotations = np.where(heatmap_data.values == 0, '', 
                                np.round(heatmap_data.values, 0).astype(int).astype(str))

        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values,
                x=[str(int(col)) for col in heatmap_data.columns],  # Convert to integer strings
                y=month_labels,
                colorscale='RdYlGn',
                text=text_annotations,
                texttemplate='%{text}',
                textfont={"size": 10},
                showscale=False,  # This removes the colorbar
                hoverongaps=False  # Don't show hover for empty cells
            ),
            row=4, col=2
        )
        
        # Update layout with title and formatting
        fig.update_layout(
            title=f'{strategy.name} - Backtest Analysis',
            showlegend=False,
            height=1600,
            width=1200,
        )
        
        # Hide colour bar for heatmap
        fig.update_coloraxes(showscale=False)
        
        # Update axes labels for each subplot
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative P&L ($)", row=1, col=1)
        fig.update_xaxes(title_text="P&L ($) / Log Return (%)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_yaxes(title_text="Win Rate (%)", row=2, col=1)
        fig.update_xaxes(title_text="Day of Week", row=2, col=2)
        fig.update_yaxes(title_text="Avg Log Return (%)", row=2, col=2)
        fig.update_xaxes(title_text="Price Change (%)", row=3, col=1)
        fig.update_yaxes(title_text="P&L ($)", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=2)
        fig.update_yaxes(title_text="Log Return (%)", row=3, col=2)
        fig.update_yaxes(title_text="Volatility (%)", row=3, col=2, secondary_y=True)
        fig.update_xaxes(title_text="Entry Price ($)", row=4, col=1)
        fig.update_yaxes(title_text="Expiry Price ($)", row=4, col=1)
        fig.update_xaxes(title_text="Year", row=4, col=2)
        fig.update_yaxes(title_text="Month", row=4, col=2)
        
        return fig
