import streamlit as st
import numpy as np
import pandas as pd 
import pickle,os
import matplotlib.pyplot as plt

def load_model():
    with open("vehicle_model.pkl", "rb") as f:
        return pickle.load(f)

st.set_page_config(
    page_title="ML Model Showcase",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)