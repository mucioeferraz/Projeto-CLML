import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from warnings import filterwarnings; filterwarnings('ignore') # retira avisos
import gc
from statsmodels.tsa.stattools import adfuller

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



def adf(y, uf, bioma):
    result = adfuller(y, autolag='AIC')
        
    if result[1] > .05:
        status = 'NÃO É ESTACIONARIA'
    else:
        status = 'É ESTACIONARIA'

    return uf, bioma, result[0], result[1], status, result[4]