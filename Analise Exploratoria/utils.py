import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from warnings import filterwarnings; filterwarnings('ignore') # retira avisos
from torch.nn import Embedding
import torch
from sklearn.preprocessing import LabelEncoder
from pmdarima.arima import auto_arima
import gc

def df_import(year_range):

    # PERCORRER TODOS OS ANOS    
    for i, year in enumerate(range(year_range[0], year_range[1])):
        
        # SE FOR PRIMEIRO ARQUIVO CRIAR NOVO DF, SE NÃO, CONCATENAR COM DADOS ANTERIORES
        if i == 0:
            df = pd.read_csv(f'../Dados/focos_qmd_inpe_{year}.csv')
        else:
            df = pd.concat([df, pd.read_csv(f'../Dados/focos_qmd_inpe_{year}.csv')], axis=0)
    
    # RESET INDEX DO DATAFRAME
    df.reset_index(drop=True, inplace=True)

    return df



def df_describe(df):
    # VALORES NULOS
    print('VALORES NULOS')
    print(df.isnull().sum(), end='\n\n')

    # TIPO DADOS
    print('DTYPES')
    print(df.dtypes, end='\n\n')

    # TAMANHO DATA-FRAME
    print('SHAPE')
    print(df.shape)

    # DESCRIBE DATA-FRAME
    print('DESCRIBE')
    display(df.describe())



def outliers_dbscan(df, eps=0.5, min_samples=5):

    # NIVEL DO GROUPO DE ITENS FILTRADOS QUE SERÁ APLICADO NO DBSCAN
    df['UF_BIOMA'] = df['Estado'] + '_' + df['Bioma']

    # SE EXISTIR COLUNA OUTLIERS, REMOVER DO DATAFRAME E CALCULAR NOVAMENTE.
    try:
        df.drop('outliers', axis=1, inplace=True)
    except:
        pass

    # CRIAR TABELA SEM DADOS
    outliers = pd.DataFrame(columns=['outliers'])

    # PERCORRER NIVEIS
    for nivel in df['UF_BIOMA'].unique().tolist():
        
        # FILTRAR MESMO NIVEL DE DADOS
        df_temp = df.loc[df['UF_BIOMA'] == nivel]

        # FILTRAR COLUNAS
        df_temp = df_temp[['DiaSemChuva','Precipitacao','RiscoFogo','FRP','dia','mes','ano','semana']]

        # ENCONDER
        for col in df_temp.columns.unique().tolist():
            df_temp[col] = MinMaxScaler().fit_transform(X=df_temp[[col]])

        # IDENTIFICAR OUTLIERS
        df_temp['outliers'] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X=df_temp) < 0
        
        # SE OUTLIER: 1 // SE NÃO: 0
        df_temp['outliers'] = df_temp['outliers'].astype(int)

        # CONCATENAR NA VARIAVEL OUTLIERS VALORES PREDICT DBSCAN (SE NÃO EXITIR DADOS NA VARIAVEL CRIAR, VARIAVEL)
        outliers = df_temp['outliers'] if len(outliers)==0 else pd.concat([outliers, df_temp['outliers']], axis=0)
        
        # LIBERAR MEMORIA
        del df_temp
        gc.collect()

    # JOIN INFOS NA BASE ORIGINAL PELO INDEX DAS BASES
    df = df.join(other=outliers, how='left')

    return df


def rotulos(grafico):

    for container in grafico.containers:
        grafico.bar_label(container, label_type="edge", color="black",
                    padding=6,
                    fontsize=9,
                    bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white', 'edgecolor': 'black'})



def embedding_data(df):

    # EMBEDDING ESTADO
    emb = Embedding(num_embeddings=len(df['Estado'].unique().tolist()), embedding_dim=1)
    uf_tensors = torch.tensor(LabelEncoder().fit_transform(df['Estado']))
    df['Estado_emb'] = emb(uf_tensors).tolist()
    df['Estado_emb'] = df['Estado_emb'].apply(lambda x: x[0])

    # EMBEDDING BIOMA
    emb = Embedding(num_embeddings=len(df['Bioma'].unique().tolist()), embedding_dim=1)
    bioma_tensors = torch.tensor(LabelEncoder().fit_transform(df['Bioma']))
    df['Bioma_emb'] = emb(bioma_tensors).tolist()
    df['Bioma_emb'] = df['Bioma_emb'].apply(lambda x: x[0])

    return df



def df_train_test_split(df, filtro, perc=.75):

    # FILTRAR DADOS
    df_temp = df\
        .loc[df['UF_BIOMA'] == filtro]\
        .drop(['Estado','Bioma','UF_BIOMA'], axis=1)\
        .sort_index(ascending=True)


    # TRAIN TEST SPLIT
    n = round(len(df_temp)*perc)
    train = df_temp.iloc[:n]
    test = df_temp[n:]


    # X E Y - TRAIN 
    x_train = train.drop('RiscoFogo', axis=1).values
    y_train = train['RiscoFogo'].values


    # X E Y - TEST 
    x_test = test.drop('RiscoFogo', axis=1).values
    y_test = test['RiscoFogo'].values

    # X E Y_TRUE TRAIN
    df_train = train.reset_index()
    df_train['UF_BIOMA'] = filtro

    # X E Y_TRUE DO TESTE
    df_test = test.reset_index()
    df_test['UF_BIOMA'] = filtro


    return x_train, y_train, x_test, y_test, df_train, df_test



def predict(x,y):
    
    stepwise_model = auto_arima(
         y=y
        ,X=x
        ,start_p=1
        ,start_q=1
        ,max_p=6
        ,max_q=6
        ,start_P=0
        ,seasonal=True
        ,m=int(round(len(x)/5))
        ,d=1
        ,D=1
        ,trace=False
        ,error_action='ignore'
        ,suppress_warnings=True
        ,stepwise=True)
    
    print(f'AIC: {stepwise_model.aic()}')

    return stepwise_model