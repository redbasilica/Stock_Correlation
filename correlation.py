import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from itertools import combinations
import io
import os
import json


def read_tickers_from_file():
    """Read tickers from tickers.txt file in working directory."""
    try:
        with open('tickers.txt', 'r') as f:
            tickers = []
            for line in f:
                ticker = line.strip()
                if ticker:
                    # Add .IS suffix for Istanbul Stock Exchange if not present
                    if not ticker.endswith('.IS'):
                        ticker = f"{ticker}.IS"
                    tickers.append(ticker)
        return tickers
    except FileNotFoundError:
        st.error("❌ tickers.txt file not found in working directory")
        return None
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        return None


def get_data_filename():
    """Get filename for today's data."""
    return f"{datetime.now().strftime('%Y-%m-%d')}.json"


def save_data_to_file(data, filename):
    """Save fetched data to JSON file."""
    try:
        # Convert pandas Series to dict for JSON serialization
        data_dict = {}
        for ticker, series in data.items():
            data_dict[ticker] = {
                'dates': series.index.strftime('%Y-%m-%d').tolist(),
                'values': series.tolist()
            }

        with open(filename, 'w') as f:
            json.dump(data_dict, f, indent=2)
        return True
    except Exception as e:
        st.error(f"❌ Error saving data: {e}")
        return False


def load_data_from_file(filename):
    """Load data from JSON file."""
    try:
        with open(filename, 'r') as f:
            data_dict = json.load(f)

        # Convert back to pandas Series
        data = {}
        for ticker, values in data_dict.items():
            dates = pd.to_datetime(values['dates'])
            data[ticker] = pd.Series(values['values'], index=dates)

        return data
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None


def get_historical_data(tickers, num_days, progress_bar, status_text):
    """Fetch historical closing prices for all tickers."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days + 10)  # Extra days for trading days

    data = {}
    failed_tickers = []

    for i, ticker in enumerate(tickers, 1):
        progress_bar.progress(i / len(tickers))
        status_text.text(f"Fetching data: {i}/{len(tickers)} tickers...")

        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)

            if not hist.empty and len(hist) >= 2:
                closing_prices = hist['Close'].tail(num_days)
                if len(closing_prices) >= 2:
                    data[ticker] = closing_prices
                else:
                    failed_tickers.append(ticker)
            else:
                failed_tickers.append(ticker)
        except Exception:
            failed_tickers.append(ticker)

    return data, failed_tickers


def calculate_price_change(data):
    """Calculate percentage change for each ticker over the period."""
    changes = {}
    for ticker, prices in data.items():
        if len(prices) >= 2:
            first_price = prices.iloc[0]
            last_price = prices.iloc[-1]
            if first_price != 0:
                change_percent = ((last_price - first_price) / first_price) * 100
                changes[ticker] = change_percent
    return changes


def filter_tickers_by_change(data, min_change, max_change):
    """Filter tickers based on price change percentage."""
    changes = calculate_price_change(data)
    filtered_data = {}

    for ticker, change in changes.items():
        if min_change <= change <= max_change:
            filtered_data[ticker] = data[ticker]

    return filtered_data, changes


def calculate_correlations(data, min_correlation, progress_bar, status_text):
    """Calculate correlation coefficients for all ticker pairs."""
    tickers = list(data.keys())
    correlations = []

    # Filter out tickers with zero variance
    valid_tickers = []
    for ticker in tickers:
        if data[ticker].std() > 0:
            valid_tickers.append(ticker)

    if len(valid_tickers) < len(tickers):
        status_text.text(f"Filtered out {len(tickers) - len(valid_tickers)} tickers with constant prices.")

    pairs_list = list(combinations(valid_tickers, 2))
    total_pairs = len(pairs_list)

    for idx, (ticker1, ticker2) in enumerate(pairs_list, 1):
        if idx % 1000 == 0 or idx == total_pairs:
            progress_bar.progress(idx / total_pairs)
            status_text.text(f"Calculating correlations: {idx}/{total_pairs} pairs...")

        try:
            df = pd.DataFrame({
                ticker1: data[ticker1],
                ticker2: data[ticker2]
            }).dropna()

            if len(df) >= 2:
                if df[ticker1].std() > 0 and df[ticker2].std() > 0:
                    corr = df[ticker1].corr(df[ticker2])

                    # Check if correlation is valid (not NaN) and positive
                    if pd.notna(corr) and corr >= min_correlation / 100:
                        correlations.append({
                            'ticker1': ticker1,
                            'ticker2': ticker2,
                            'correlation': corr
                        })
        except Exception:
            continue

    return correlations


def main():
    st.set_page_config(
        page_title="Stock Correlation Analyzer",
        page_icon="📈",
        layout="wide"
    )

    st.title("📈 Stock Correlation Analyzer")
    st.markdown("### Turkish Stock Market (Borsa Istanbul)")
    st.markdown("---")

    # Sidebar for inputs
    with st.sidebar:
        st.header("⚙️ Configuration")

        num_days = st.number_input(
            "Number of days to retrieve",
            min_value=1,
            max_value=365,
            value=7,
            help="Number of days of historical data to analyze"
        )

        st.markdown("---")

        st.subheader("Price Change Filter")

        col1, col2 = st.columns(2)
        with col1:
            min_change = st.number_input(
                "Min change %",
                min_value=-100.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
                help="Minimum price change percentage"
            )
        with col2:
            max_change = st.number_input(
                "Max change %",
                min_value=-100.0,
                max_value=100.0,
                value=100.0,
                step=1.0,
                help="Maximum price change percentage"
            )

        st.markdown("---")

        top_x = st.slider(
            "Number of top pairs to display (x)",
            min_value=1,
            max_value=20,
            value=10,
            help="Select how many top correlated pairs to show"
        )

        min_corr_percent = st.slider(
            "Minimum correlation percentage (y)",
            min_value=1,
            max_value=100,
            value=70,
            help="Only show pairs with correlation above this threshold"
        )

        st.markdown("---")

        analyze_button = st.button("🚀 Analyze Correlations", type="primary", use_container_width=True)

    # Main content area
    # Read tickers from file
    tickers = read_tickers_from_file()
    if not tickers:
        st.info("📄 Please ensure tickers.txt file exists in the working directory")
        st.markdown("""
        ### tickers.txt format:
        ```
        A1CAP
        THYAO
        GARAN
        ISCTR
        ```

        **Note:** The .IS suffix will be added automatically for Istanbul Stock Exchange tickers.
        """)
        return

    st.success(f"✅ Loaded {len(tickers)} tickers from tickers.txt")

    with st.expander("📋 View loaded tickers"):
        cols = st.columns(4)
        for idx, ticker in enumerate(tickers):
            cols[idx % 4].text(ticker)

    if analyze_button:
        st.markdown("---")
        st.subheader("📊 Analysis Results")

        # Check if data file exists for today
        data_filename = get_data_filename()
        data = None

        if os.path.exists(data_filename):
            st.info(f"📂 Found existing data file: {data_filename}")
            with st.spinner("Loading data from file..."):
                data = load_data_from_file(data_filename)

            if data:
                st.success(f"✅ Loaded data for {len(data)} tickers from file")

        # If no data loaded, fetch from server
        if data is None:
            st.info("🌐 No data file found. Fetching from server...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner("Fetching stock data..."):
                data, failed_tickers = get_historical_data(tickers, num_days, progress_bar, status_text)

            progress_bar.empty()
            status_text.empty()

            if not data:
                st.error("❌ No valid data fetched for any ticker")
                return

            st.success(f"✅ Successfully fetched data for {len(data)} tickers")

            # Save data to file
            if save_data_to_file(data, data_filename):
                st.success(f"💾 Data saved to {data_filename}")

            if failed_tickers:
                with st.expander(f"⚠️ Failed to fetch data for {len(failed_tickers)} tickers"):
                    st.write(", ".join(failed_tickers))

        # Filter tickers by price change
        st.markdown("---")
        st.subheader("🔍 Filtering by Price Change")

        filtered_data, all_changes = filter_tickers_by_change(data, min_change, max_change)

        st.info(f"📊 Filtered: {len(filtered_data)} tickers with price change between {min_change}% and {max_change}%")

        with st.expander("📈 View price changes for all tickers"):
            changes_df = pd.DataFrame([
                {'Ticker': ticker, 'Price Change %': f"{change:.2f}%"}
                for ticker, change in sorted(all_changes.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(changes_df, use_container_width=True, hide_index=True)

        if len(filtered_data) < 2:
            st.warning("⚠️ Not enough tickers after filtering. Adjust the price change range.")
            return

        # Calculate correlations
        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner("Calculating correlations..."):
            correlations = calculate_correlations(filtered_data, min_corr_percent, progress_bar, status_text)

        progress_bar.empty()
        status_text.empty()

        # Display results
        if not correlations:
            st.warning(f"⚠️ No pairs found with correlation ≥ {min_corr_percent}%")
            return

        # Sort by correlation value (descending)
        sorted_corr = sorted(correlations, key=lambda x: x['correlation'], reverse=True)

        st.success(f"✅ Found {len(correlations)} pairs with correlation ≥ {min_corr_percent}%")

        # Create DataFrame for display
        display_data = []
        for i, pair in enumerate(sorted_corr[:top_x], 1):
            ticker1_change = all_changes.get(pair['ticker1'], 0)
            ticker2_change = all_changes.get(pair['ticker2'], 0)

            display_data.append({
                'Rank': i,
                'Ticker 1': pair['ticker1'],
                'Change 1 %': f"{ticker1_change:.2f}%",
                'Ticker 2': pair['ticker2'],
                'Change 2 %': f"{ticker2_change:.2f}%",
                'Correlation': f"{pair['correlation']:.4f}",
                'Correlation %': f"{pair['correlation'] * 100:.2f}%"
            })

        df_display = pd.DataFrame(display_data)

        # Display as table
        st.markdown(f"### Top {min(top_x, len(sorted_corr))} Positively Correlated Pairs")
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True
        )

        # Download button for full results
        csv_buffer = io.StringIO()
        full_df = pd.DataFrame([
            {
                'Ticker 1': pair['ticker1'],
                'Change 1 %': all_changes.get(pair['ticker1'], 0),
                'Ticker 2': pair['ticker2'],
                'Change 2 %': all_changes.get(pair['ticker2'], 0),
                'Correlation': pair['correlation'],
                'Correlation %': pair['correlation'] * 100
            }
            for pair in sorted_corr
        ])
        full_df.to_csv(csv_buffer, index=False)

        st.download_button(
            label="📥 Download Full Results (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"correlations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tickers Analyzed", len(filtered_data))
        with col2:
            st.metric("Total Pairs Found", len(correlations))
        with col3:
            avg_corr = sum(c['correlation'] for c in correlations) / len(correlations)
            st.metric("Average Correlation", f"{avg_corr:.2%}")
        with col4:
            max_corr = sorted_corr[0]['correlation']
            st.metric("Highest Correlation", f"{max_corr:.2%}")


if __name__ == "__main__":
    main()
