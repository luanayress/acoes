import numpy as np
from datetime import date
import pandas as pd
import streamlit as st
import random
import plotly.graph_objects as go
import yfinance as yf

acao_escolhida = st.sidebar.text_input("Digite a Ação", value="AAPL")

st.write('teste')
