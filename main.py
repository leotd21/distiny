
import streamlit as st
from app import controller

def main():
    st.set_page_config(page_title="Synthetic Data Generator", layout="centered")
    st.title("🧪 Synthetic Tabular Data Generator")
    controller.run()

if __name__ == "__main__":
    main()
