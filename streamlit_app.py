import streamlit as st
from datetime import datetime

from utils.data_loader import get_available_files, get_all_stock_codes, get_date_from_filename
from ui.market_summary import page_market_summary
from ui.trend_analysis import page_trend_analysis
from ui.strategy_analysis import page_strategy_analysis
from ui.price_prediction import page_price_prediction
from ui.comprehensive_trend_analysis import page_comprehensive_trend_analysis

def main():
    """
    Main function to run the Streamlit application.
    This function sets up the page configuration and sidebar navigation,
    and then routes to the appropriate page function based on user selection.
    """
    st.set_page_config(layout="wide")
    st.title("IDX Stock Analysis Dashboard")

    # --- Data Loading ---
    # Load essential data once and pass it to the page functions
    files = get_available_files()
    if not files:
        st.error("No data files found in the 'data' directory. Please run the scraper first.")
        return

    stock_codes = get_all_stock_codes()
    available_dates = [get_date_from_filename(f) for f in files]

    # --- Sidebar Navigation ---
    st.sidebar.title("Navigation")
    page_options = {
        "Daily Market Summary": page_market_summary,
        "Stock Trend Analysis": page_trend_analysis,
        "Strategy Analysis": page_strategy_analysis,
        "Price Prediction": page_price_prediction,
        "Comprehensive Trend Analysis": page_comprehensive_trend_analysis,
    }
    selected_page = st.sidebar.radio("Choose a page", list(page_options.keys()))

    # --- Page Routing ---
    if selected_page in ["Daily Market Summary"]:
        page_options[selected_page](available_dates)
    elif selected_page in ["Stock Trend Analysis", "Strategy Analysis", "Price Prediction"]:
        page_options[selected_page](stock_codes, available_dates)
    else:
        page_options[selected_page]()

if __name__ == "__main__":
    main()
