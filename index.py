# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import mlrose
import locale
import sys
from pandas_datareader import data
from datetime import datetime
 
start_time = datetime.now()
print('Start Processing ' + str(start_time))

now = datetime.now()

valor = 100000
qtyears = 5

stock = ['PETR4.SA','LEVE3.SA','IRBR3.SA','KISU11.SA','HGLG11.SA','ENGI4.SA','ITSA4.SA','PETZ3.SA','STBP3.SA']
#stock = ['JHSF3.SA','WEGE3.SA','CSNA3.SA','EZTC3.SA','PETZ3.SA','ENBR3.SA','MDIA3.SA','SQIA3.SA','ITSA4.SA','FLRY3.SA','WIZS3.SA']
yahoodata = data.DataReader(stock, data_source='yahoo', start=str(now.year - qtyears) + '-01-01')['Adj Close']
yahoodata = yahoodata.reset_index().replace(np.nan, 1)

def moeda(v):
    locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
    v = locale.currency(v, grouping=True, symbol=None)
    return ('R$%s' % v)

# data = pd.read_csv('acoes.csv')
qtcolunas = len(yahoodata.columns)-1


taxaselic2015 = 12.75
taxaselic2016 = 14.25
taxaselic2017 = 12.25
taxaselic2018 = 6.50
taxaselic2019 = 5.0
taxaselic2020 = 2.75
taxaselic2021 = 4.25
taxaselichistorico = np.array([12.75,14.25,12.25,6.50,5.0,2.75,4.25])

semrisco = taxaselichistorico.mean()/100

def isinf(value):
    return abs(value) == float("infinity")

def alocacao_ativos(dataset, dinheiro_total, seed = 0, melhores_pesos = []):
    dataset = yahoodata.copy()
    if seed != 0:
       np.random.seed(seed)

    if len(melhores_pesos) > 0:
        pesos = melhores_pesos
    else:  
        pesos = np.random.random(len(dataset.columns) - 1)
        pesos = pesos / pesos.sum()

    colunas = dataset.columns[1:]
    for i in colunas:
        dataset[i] = (dataset[i] / dataset[i][0])

    for i, acao in enumerate(dataset.columns[1:]):
        dataset[acao] = dataset[acao] * pesos[i] * dinheiro_total
  
    dataset['soma valor'] = dataset.sum(axis = 1)
    datas = dataset['Date']
    dataset.drop(labels = ['Date'], axis = 1, inplace = True)
    dataset['taxa retorno'] = 0.0


    for i in range(1, len(dataset)):
        dataset['taxa retorno'][i] = ((dataset['soma valor'][i] / dataset['soma valor'][i - 1]) - 1) * 100
    
    acoes_pesos = pd.DataFrame(data = {'Ações': colunas, 'Pesos': pesos * 100})
    
    return dataset, datas, acoes_pesos, dataset.loc[len(dataset) - 1]['soma valor']



def alocacao_portfolio(dataset, dinheiro_total, sem_risco, repeticoes):
    dataset = yahoodata.copy()
    dataset_original = dataset.copy()

    lista_retorno_esperado = []
    lista_volatilidade_esperada = []
    lista_sharpe_ratio = []

    melhor_sharpe_ratio = 1 - sys.maxsize
    melhores_pesos = np.empty
    melhor_volatilidade = 0
    melhor_retorno = 0
  
    for _ in range(repeticoes):
        pesos = np.random.random(len(dataset.columns) - 1)
        pesos = pesos / pesos.sum()

    for i in dataset.columns[1:]:
        dataset[i] = dataset[i] / dataset[i][0]

    for i, acao in enumerate(dataset.columns[1:]):
        dataset[acao] = dataset[acao] * pesos[i] * dinheiro_total

    dataset.drop(labels = ['Date'], axis = 1, inplace=True)

    retorno_carteira = np.log(dataset / dataset.shift(1))
    matriz_covariancia = retorno_carteira.cov()

    dataset['soma valor'] = dataset.sum(axis = 1)
    dataset['taxa retorno'] = 0.0

    for i in range(1, len(dataset)):
        dataset['taxa retorno'][i] = np.log(dataset['soma valor'][i] / dataset['soma valor'][i - 1])

    sharpe_ratio = (dataset['taxa retorno'].mean() - sem_risco) / dataset['taxa retorno'].std() * np.sqrt(246)
    if sharpe_ratio < 1: 
        sharpe_ratio = 1

    retorno_esperado = np.sum(dataset['taxa retorno'].mean() * pesos) * 246
    volatilidade_esperada = np.sqrt(np.dot(pesos, np.dot(matriz_covariancia * 246, pesos)))
    
    sharpe_ratio = (retorno_esperado - sem_risco) / volatilidade_esperada
 
    if sharpe_ratio > melhor_sharpe_ratio:
      melhor_sharpe_ratio = sharpe_ratio
      melhores_pesos = pesos
      melhor_volatilidade = volatilidade_esperada
      melhor_retorno = retorno_esperado

    lista_retorno_esperado.append(retorno_esperado)
    lista_volatilidade_esperada.append(volatilidade_esperada)
    lista_sharpe_ratio.append(sharpe_ratio)
     
    dataset = dataset_original.copy()

    return melhor_sharpe_ratio, melhores_pesos, lista_retorno_esperado, lista_volatilidade_esperada, lista_sharpe_ratio, melhor_volatilidade, melhor_retorno



def fitness_function(solucao):
    dataset = yahoodata.copy()

    if solucao.sum() < 1:
        solucaosum = 1
    else:
        solucaosum = solucao.sum()
    
    pesos = solucao / solucaosum
 

    for i in dataset.columns[1:]:
        dataset[i] = (dataset[i] / dataset[i][0])

    for i, acao in enumerate(dataset.columns[1:]):
        dataset[acao] = dataset[acao] * pesos[i] * valor

    dataset.drop(labels = ['Date'], axis = 1, inplace=True)

    dataset['soma valor'] = dataset.sum(axis = 1)
    dataset['taxa retorno'] = 0.0

    for i in range(1, len(dataset)):
        dataset['taxa retorno'][i] = ((dataset['soma valor'][i] / dataset['soma valor'][i - 1]) - 1) * 100

    sharpe_ratio = (dataset['taxa retorno'].mean() - semrisco) / dataset['taxa retorno'].std() * np.sqrt(246)
    return sharpe_ratio

np.random.seed(10)
pesos = np.random.random(qtcolunas)
pesos = pesos / pesos.sum()

print('\n\nInvestimento: ' + str(moeda(valor)) +' Período: '+str(qtyears)+' anos\n' )
print('######## OPTIMIZATION RANDOM #############')
print('Processing random weights...')

def view_location(solucao):
    colunas = yahoodata.columns[1:]
    for i in range(len(solucao)):
        print(colunas[i], str(round(solucao[i] * 100,0))+'%', str(moeda(valor * solucao[i]))  )
        
view_location(pesos)
_, _, _, soma_valor = alocacao_ativos(yahoodata, valor, melhores_pesos=pesos)
print('Retorno: ' + str(moeda(soma_valor)))


print('\n\n######## OPTIMIZATION MARKOWITZ #############')
print('Processing weights...')
sharpe_ratio, melhores_pesos2, ls_retorno, ls_volatilidade, ls_sharpe_ratio, melhor_volatilidade, melhor_retorno = alocacao_portfolio(yahoodata, valor, taxaselichistorico.mean() / 100, 1000)
_, _, acoes_pesos, soma_valor = alocacao_ativos(yahoodata, valor, melhores_pesos=melhores_pesos2)
view_location(melhores_pesos2)
print('Retorno: ' + str(moeda(soma_valor)))

print('\n\n######## PORTFOLIO HILL CLIMB #############')
print('Processing fitness...')
fitness = mlrose.CustomFitness(fitness_function)
problema_maximizacao = mlrose.ContinuousOpt(length=len(stock), fitness_fn=fitness, maximize = True, min_val = 0, max_val = 1)
melhor_solucao, melhor_custo = mlrose.hill_climb(problema_maximizacao, random_state = 1)
melhor_solucao = melhor_solucao / melhor_solucao.sum()
view_location(melhor_solucao)
_, _, _, soma_valor = alocacao_ativos(yahoodata, valor, melhores_pesos=melhor_solucao)
print('Retorno: ' + str(moeda(soma_valor)))

print('\n\n######## PORTFOLIO SIMULATED ANNEALING #############')
print('Processing fitness...')
melhor_solucao, melhor_custo = mlrose.simulated_annealing(problema_maximizacao, random_state = 1)
melhor_solucao = melhor_solucao / melhor_solucao.sum()
view_location(melhor_solucao)
_, _, _, soma_valor = alocacao_ativos(yahoodata, valor, melhores_pesos=melhor_solucao)
print('Retorno: ' + str(moeda(soma_valor)))


print('\n\n######## PORTFOLIO GENETIC AGORITM #############')
print('Processing fitness...')
problema_maximizacao_ag = mlrose.ContinuousOpt(length = len(stock), fitness_fn = fitness, maximize = True, min_val = 0.1, max_val = 1)
melhor_solucao, melhor_custo = mlrose.genetic_alg(problema_maximizacao_ag, random_state = 1)
melhor_solucao = melhor_solucao / melhor_solucao.sum()
view_location(melhor_solucao)
_, _, _, soma_valor = alocacao_ativos(yahoodata, valor, melhores_pesos=melhor_solucao)
print('Retorno: ' + str(moeda(soma_valor)))

print('\n\nEnd Processing ' + str(datetime.now()))
print('Time Processing ' + str(datetime.now() - start_time))
print('End Process ;)')
