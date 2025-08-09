import streamlit as st
import pandas as pd

from stock_analyzer import analyze_stock_data

def page_comprehensive_trend_analysis():
    """
    Renders the 'Comprehensive Trend Analysis' page, allowing users to run and filter
    a full analysis of all stocks.
    """
    st.title("Comprehensive Stock Trend Analysis")
    st.write(
        "This section provides a detailed trend analysis for all available stocks "
        "based on SMA, MACD, and ADX indicators. The analysis may take a few moments to run."
    )

    if st.button("Run Comprehensive Analysis"):
        with st.spinner("Analyzing all stock data... This might take a while."):
            analysis_results = analyze_stock_data()

            if analysis_results is not None and not analysis_results.empty:
                st.subheader("Latest Stock Analysis Results")

                # --- Filtering ---
                st.sidebar.subheader("Filter Analysis Results")

                all_trends = ["All"] + list(analysis_results['Trend'].unique())
                all_strengths = ["All"] + list(analysis_results['Strength'].unique())
                all_phases = ["All"] + list(analysis_results['Phase'].unique())
                all_signals = ["All"] + list(analysis_results['Signal'].unique())

                selected_trend = st.sidebar.selectbox("Filter by Trend", all_trends)
                selected_strength = st.sidebar.selectbox("Filter by Strength", all_strengths)
                selected_phase = st.sidebar.selectbox("Filter by Phase", all_phases)
                selected_signal = st.sidebar.selectbox("Filter by Signal", all_signals)

                # Apply filters
                filtered_df = analysis_results.copy()
                if selected_trend != "All":
                    filtered_df = filtered_df[filtered_df['Trend'] == selected_trend]
                if selected_strength != "All":
                    filtered_df = filtered_df[filtered_df['Strength'] == selected_strength]
                if selected_phase != "All":
                    filtered_df = filtered_df[filtered_df['Phase'] == selected_phase]
                if selected_signal != "All":
                    filtered_df = filtered_df[filtered_df['Signal'] == selected_signal]

                st.dataframe(filtered_df)

                # --- Download Button ---
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Filtered Results as CSV",
                    data=csv,
                    file_name="filtered_stock_analysis.csv",
                    mime="text/csv",
                )

                # Store results in session state to avoid re-running on filter change
                st.session_state['analysis_results'] = analysis_results

            else:
                st.warning("No analysis results were generated. Ensure data files are available.")

    # If results are already in session state, just display and filter them
    elif 'analysis_results' in st.session_state:
        st.subheader("Latest Stock Analysis Results")
        analysis_results = st.session_state['analysis_results']

        # --- Filtering (repeated for interactivity without re-running) ---
        st.sidebar.subheader("Filter Analysis Results")
        all_trends = ["All"] + list(analysis_results['Trend'].unique())
        # ... (rest of the filter widgets) ...

        # (Apply filtering logic as above)
        # ...

        st.dataframe(analysis_results) # Replace with filtered_df once widgets are added
        # ... (download button) ...
