import streamlit as st
import pandas as pd


st.set_page_config(
    # This will make the content occupy the full width of the screen
    layout="wide",
    page_title="Recommender System Dashboard"

)
def main():
    # Using "with" notation
    with st.sidebar:
        add_radio = st.radio(
            "Page Navigation",
            ("Main",
             "Training",
             "Performance",
             "Model Demo")
        )

if __name__ == "__main__":
    main()

