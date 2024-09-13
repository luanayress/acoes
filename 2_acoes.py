from datetime import date, timedelta
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st


DATA_INICIO = '2024-01-01'
DATA_FIM = date.today() - timedelta(days=15)

# Parâmetro para quantidade de dias de previsão
n_dias = 30  # Defina a quantidade de dias de previsão manualmente, pois não usamos Streamlit
#yf.download('PETR4.SA', start=DATA_INICIO, end=DATA_FIM)
# Função para pegar dados de ações brasileiras
def pegar_dados_acoes(ticker):
    try:
        df = yf.download(ticker, start=DATA_INICIO, end=DATA_FIM)
        if df.empty:
            raise Exception(f"Dados não encontrados para o símbolo: {ticker}")
        return df
    except Exception as e:
        print(f"Erro ao baixar dados para {ticker}: {e}")
        return pd.DataFrame()

# Substitua pelo símbolo da ação desejada (por exemplo, 'PETR4.SA' para Petrobras)
acao_escolhida = 'PETR4.SA'

# Pegar os valores históricos da ação
df_valores = pegar_dados_acoes(acao_escolhida)

# Plotar o gráfico dos preços usando Matplotlib ao invés de Streamlit
plt.figure(figsize=(10, 6))
plt.plot(df_valores.index, df_valores['Close'], label='Preço de Fechamento', color='yellow')
plt.plot(df_valores.index, df_valores['Open'], label='Preço de Abertura', color='blue')
plt.title(f"Preços de Abertura e Fechamento de {acao_escolhida}")
plt.xlabel("Data")
plt.ylabel("Preço (R$)")
plt.legend()
plt.show()

# Preparar os dados para o modelo Prophet
df_treino = df_valores.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
df_treino['cap'] = df_treino['y'].max() * 1.1  # Ajuste conforme necessário

# Instanciar o modelo Prophet e adicionar as regressoras
modelo = Prophet(
    growth='logistic',
    n_changepoints=100,
    changepoint_range=0.5,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative',
    seasonality_prior_scale=2.0,
    holidays_prior_scale=5.0,
    changepoint_prior_scale=1.1,
    mcmc_samples=500,
    interval_width=0.95,
    uncertainty_samples=1500
)

# Adicionar as regressoras ao modelo
modelo.add_regressor('Open')
modelo.add_regressor('High')
modelo.add_regressor('Low')
modelo.add_regressor('Volume')

# Ajustar o modelo com os dados históricos
modelo.fit(df_treino)

# Fazer a previsão para os próximos n_dias
futuro = modelo.make_future_dataframe(periods=n_dias)

futuro['Open'] = df_treino['Open'].iloc[-1]
futuro['High'] = df_treino['High'].iloc[-1]
futuro['Low'] = df_treino['Low'].iloc[-1]
futuro['Volume'] = df_treino['Volume'].iloc[-1]
futuro['cap'] = df_treino['cap'].iloc[0]  # Use o mesmo valor de 'cap' do treinamento ou ajuste conforme necessário

previsao = modelo.predict(futuro)

# Plotar a previsão
fig_forecast = plot_plotly(modelo, previsao)
#fig_forecast.show()

# Exibir o gráfico no Streamlit
st.plotly_chart(fig_forecast, use_container_width=True)

fig_components = modelo.plot_components(previsao)
plt.show()

# Assumindo que `previsao` já foi gerada com `modelo.predict(futuro)`
# Plotando a previsão usando Matplotlib
fig, ax = plt.subplots(figsize=(10, 6))
modelo.plot(previsao, ax=ax)


# Assumindo que os dados foram preparados e o modelo Prophet foi treinado
# E que `previsao` já foi gerada com `modelo.predict(futuro)`

# Plotando a previsão usando Matplotlib
fig, ax = plt.subplots(figsize=(10, 6))
modelo.plot(previsao, ax=ax)

# Personalizar o gráfico, se necessário
ax.set_title("Previsão com Prophet")
ax.set_xlabel("Data")
ax.set_ylabel("Preço (R$)")

# Exibir o gráfico no Streamlit
st.pyplot(fig)