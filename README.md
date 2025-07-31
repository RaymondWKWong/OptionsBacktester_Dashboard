# Options Strategy Backtester Dashboard

Simple options strategy backtester over historical data with performance metrics and visualisations

## Supported Strategies

Can add additional option legs within backtest.py to create new option strategies

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
6. Analyze performance metrics and visualisations

## Performance Metrics

- Total Return & P&L
- Win Rate & Trade Count
- Sharpe Ratio & Max Drawdown
- Profit Factor & Volatility
- Monthly performance breakdown
- Rolling performance analysis

## Visualisations

The dashboard provides 8 different analytical charts:
- Cumulative P&L vs benchmark
- Return distribution histograms
- Monthly win rate analysis
- Day-of-week performance
- Price movement correlation
- Rolling performance metrics
- Entry vs expiry price scatter
- Monthly P&L heatmap

## License

This project is for educational and research purposes.
