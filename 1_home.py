# Imports
import random
import numpy as np
from datetime import date
import pandas as pd
import yfinance as yf

DATA_INICIO = '2024-01-01'
DATA_FIM = date.today()

def pegar_dados_acoes(ticker):
    try:
        df = yf.download(ticker, start=DATA_INICIO, end=DATA_FIM)
        if df.empty:
            raise Exception(f"Dados não encontrados para o símbolo: {ticker}")
        return df
    except Exception as e:
        print(f"Erro ao baixar dados para {ticker}: {e}")
        return pd.DataFrame()

# Código adicional para o painel em Streamlit
import streamlit as st

acao_escolhida = st.sidebar.text_input("Digite a Ação", value="AAPL")

# Pegar os valores históricos da ação
df = pegar_dados_acoes(acao_escolhida).reset_index()

# Candlestick

# Vamos trabalhar com a cotação de fechamento da ação da Apple
precos = df['Close'].values

# Configuração do Q-Learning
print('\nDefinindo os Hiperparâmetros do Q-Learning...')
num_episodios = 1000
alfa = 0.1
gama = 0.99
epsilon = 0.1

# Configuração do ambiente de negociação
print('\nConfigurando o Ambiente de Negociação...')
acoes = ['comprar', 'vender', 'manter']
saldo_inicial = 1000
num_acoes_inicial = 0

# Função para executar uma ação e retornar a recompensa e o próximo estado
def executar_acao(estado, acao, saldo, num_acoes, preco):
    
    # Ação de comprar
    if acao == 0:  
        if saldo >= preco:
            num_acoes += 1
            saldo -= preco
    
    # Ação de vender
    elif acao == 1:  
        if num_acoes > 0:
            num_acoes -= 1
            saldo += preco

    # Define o lucro
    lucro = saldo + num_acoes * preco - saldo_inicial

    return (saldo, num_acoes, lucro)

# Inicializar a tabela Q
print('\nInicializando a Tabela Q...')
q_tabela = np.zeros((len(precos), len(acoes)))

# Treinamento
print('\nInicializando o Treinamento...')
for _ in range(num_episodios):
    
    # Define o saldo
    saldo = saldo_inicial
    
    # Define o número de ações
    num_acoes = num_acoes_inicial

    # Loop
    for i, preco in enumerate(precos[:-1]):
        
        estado = i

        # Escolher a ação usando a política epsilon-greedy
        if np.random.random() < epsilon:
            acao = random.choice(range(len(acoes)))
        else:
            acao = np.argmax(q_tabela[estado])

        # Executar a ação e obter a recompensa e o próximo estado
        saldo, num_acoes, lucro = executar_acao(estado, acao, saldo, num_acoes, preco)
        prox_estado = i + 1

        # Atualizar a tabela Q
        q_tabela[estado][acao] += alfa * (lucro + gama * np.max(q_tabela[prox_estado]) - q_tabela[estado][acao])

print('\nTreinamento Concluído...')

# Valores iniciais para testar o agente
saldo = saldo_inicial
num_acoes = num_acoes_inicial

print('\nExecutando o Agente...')
for i, preco in enumerate(precos[:-1]):
    estado = i
    acao = np.argmax(q_tabela[estado])
    saldo, num_acoes, _ = executar_acao(estado, acao, saldo, num_acoes, preco)
    
print('\nExecução Concluída...')

# Vendendo todas as ações no último preço
saldo += num_acoes * precos[-1]
lucro = saldo - saldo_inicial
lucro_final = round(lucro,2)

print(f"\nComeçamos a Negociação com Saldo Inicial de 1000 e Tivemos Lucro de: {lucro_final}")

# Estado atual é o último índice da lista de preços (hoje)
estado_atual = len(precos) - 1

# Verificar se o estado atual está dentro do tamanho da q_tabela
if estado_atual < len(q_tabela):
    # Buscar a ação com maior valor Q (recompensa esperada)
    acao_hoje = np.argmax(q_tabela[estado_atual])  # Melhor ação para hoje

    # Mapeando a ação para a descrição
    acoes_desc = {0: 'manter', 1: 'comprar', 2: 'vender'}  # Ajuste conforme suas definições de ações
    acao_hoje_desc = acoes_desc.get(acao_hoje, 'ação desconhecida')

    # Exibir a ação recomendada para hoje
    print(f"A ação recomendada para hoje é: {acao_hoje_desc}")
else:
    print("O estado atual está fora da tabela Q. Verifique os dados.")

# Título do painel
st.title("Agente de Negociação com Q-Learning")

# Exibir resultado final
st.subheader("Resultados da Execução")
st.write(f"Saldo final após a execução: R$ {round(saldo, 2)}")
st.write(f"Lucro final: R$ {lucro_final}")

# Ação recomendada para hoje
col1, col2 = st.columns(2)
st.subheader("Ação Recomendada para Hoje")
col1.write(f"Ação Escolhida: {acao_escolhida}")
col2.write(f"Recomendação: {acao_hoje_desc}")
