from pmdarima.arima import auto_arima
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from datetime import timedelta


def train_test_split(df, train_size=.9):

    # INDEX
    df.index = df['Data']

    # SELECT COLS X, Y
    x = df[['media_dias_sem_chuva','media_precipitacao']]
    y = df['media_risco_fogo']

    # TAMANHO 
    n = round(len(x)*train_size)

    # TRAIN SPLIT
    x_train = x[:n]
    y_train = y[:n]

    # TEST SPLIT
    x_test = x[n:]
    y_test = y[n:]

    # SE TAMANHO DO TREINO FOR MENOR QUE 365 DIAS, APLICAR SAZONALIDADE DE: TAM DF / NUMERO DE ANOS 
    m = 180 if len(df.iloc[:n]) > 180 else round(len(df.iloc[:n])/len(df['Data'].iloc[:n].apply(lambda x: x.year).unique().tolist()))

    return x_train, y_train, x_test, y_test, m



def train(df, train_size=.9, trace=True):
    
    # FUNÇÃO TRAIN TEST SPLIT
    x_train, y_train, x_test, _, m = train_test_split(df, train_size=train_size)
    
    # HIPERPARAMETROS DO MODELO
    stepwise_model = auto_arima(
         exogenous=x_train
        ,y=y_train
        ,start_p=1
        ,start_q=1
        ,max_p=3
        ,max_q=3
        ,start_P=0
        ,seasonal=True
        ,m=m
        ,d=1
        ,D=1
        ,trace=trace
        ,error_action='ignore'
        ,suppress_warnings=True
        ,stepwise=True)
        
    # IMPRIMIR AIC
    print(f'AIC: {round(stepwise_model.aic(),4)}')

    return df, stepwise_model, x_test



def test(model, df, x_test):
    
    # TESTAR MODELO
    y_pred = model.predict(n_periods=len(x_test), X=x_test)

    # ADD PREDICT NO DF FINAL
    y_predict = np.full(len(df)-len(x_test), np.nan).tolist()
    y_predict.extend(y_pred.tolist())
    df['y_pred'] = y_predict

    # AIC
    df['aic'] = model.aic()
    df['best_arima'] = str(model).strip()
    
    return df



def forecast(model, df, year=2024):

    # PREDICT ATÉ FINAL DO ANO ESPECIFICADO NA FUNÇÃO
    n_periods = (datetime(year,12,31).date() - df['Data'].max()).days

    # FORECAST
    forecast = model.predict(n_periods=n_periods, exogenous=None)
    
    # MINIMO = 0 E MAXIMO = 1
    forecast = np.min([forecast, np.full(len(forecast), 1)], axis=0)
    forecast = np.max([forecast, np.full(len(forecast), 0)], axis=0)

    return forecast



def plot_forecast(forecast, df):

    # INCLUIR DATA NAS PREVISÕES DO FORECAST
    forecast = pd.DataFrame(forecast, columns=['Forecast'])
    forecast.reset_index(drop=True, inplace=True)
    forecast.reset_index(inplace=True)

    # N + DATA MAXIMA
    forecast['index'] = forecast['index'].apply(lambda x: df['Data'].max() + timedelta(days=x+1))

    # RENOMEAR PARA DATA
    forecast.rename(columns={'index':'Data'}, inplace=True)
    forecast.index = forecast['Data']

    # TITULO QUE SERA PLOTADO NO GRAFICO
    uf = df['Estado'][df['Estado'].notnull()].unique().tolist()[0].upper()
    bioma = df['Bioma'][df['Bioma'].notnull()].unique().tolist()[0].upper()
    uf_bioma = df['uf_bioma'][df['uf_bioma'].notnull()].unique().tolist()[0].upper()

    # CONCATENAR BASE COM FORECAST E PREENCHER DADOS DO FORECAST
    df = pd.concat([df, forecast], axis=0)
    df['Estado'] = uf
    df['Bioma'] = bioma
    df['uf_bioma'] = uf_bioma
    df['aic'].fillna(np.max(df['aic']), inplace=True)
    df['best_arima'].fillna(df['best_arima'].unique().tolist()[0], inplace=True)

    # PLOTAR DADOS
    df.plot(y=['media_risco_fogo','y_pred','Forecast'], label=['True','Test','Forecast'])
    plt.legend()
    plt.title(f'FORECAST: {uf} - {bioma}')
    plt.show()

    return df