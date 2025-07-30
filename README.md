# Options Strategy Backtester Dashboard

A comprehensive Streamlit web application for backtesting various options strategies on cryptocurrency data.

## Features

- **Interactive Strategy Selection**: Choose from 6 common options strategies
- **Strategy Visualization**: See payoff diagrams for each strategy
- **Customizable Parameters**: Adjust strike prices, premiums, and positions
- **Comprehensive Backtesting**: Test strategies over historical data
- **Rich Analytics**: Detailed performance metrics and visualizations
- **Beautiful UI**: Clean, professional interface with intuitive controls

## Supported Strategies

1. **Covered Call** - Long stock + Short call
2. **Cash Secured Put** - Short put with cash backing
3. **Long Straddle** - Long call + Long put at same strike
4. **Iron Condor** - Short put spread + Short call spread
5. **Bull Call Spread** - Long call + Short call at higher strike
6. **Long Strangle** - Long OTM call + Long OTM put

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Select your desired options strategy from the sidebar
2. View the strategy payoff diagram
3. Adjust strategy parameters (strikes, premiums, etc.)
4. Set backtesting parameters (date range, frequency)
5. Click "Run Backtest" to see results
6. Analyze performance metrics and visualizations

## Performance Metrics

- Total Return & P&L
- Win Rate & Trade Count
- Sharpe Ratio & Max Drawdown
- Profit Factor & Volatility
- Monthly performance breakdown
- Rolling performance analysis

## Visualizations

The dashboard provides 8 different analytical charts:
- Cumulative P&L vs benchmark
- Return distribution histograms
- Monthly win rate analysis
- Day-of-week performance
- Price movement correlation
- Rolling performance metrics
- Entry vs expiry price scatter
- Monthly P&L heatmap

## Data

Currently uses simulated cryptocurrency price data for demonstration. Can be easily adapted to use real market data from various sources.

## License

MIT License - Feel free to use and modify as needed.

## Contributing

Pull requests welcome! Please feel free to submit improvements or bug fixes.
