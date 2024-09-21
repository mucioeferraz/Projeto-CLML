import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px
import decorators


@decorators.resize(25,5)
@decorators.show()
def daily_plot(df, y, color):
    sns.lineplot(data=df, x='Data', y=y, color=color)
    sns.lineplot(data=df, x='Data', y=y.replace('media_', 'media_movel_'), color='black')
    plt.title(y.strip().replace('_',' ').upper())



@decorators.resize(8,5)
@decorators.show()
def correlacao(df, method):
    sns.heatmap(df.corr(method=method, numeric_only=True), annot=True, fmt='.1f', cmap='binary')
    plt.title(method.upper())



@decorators.resize(20,8)
@decorators.rotulos
def perc_outliers(df, nivel):

    outliers = df\
                .groupby([nivel], as_index=False)\
                .agg(count_outliers=('outliers','sum'), count_lines=('outliers', 'count'))

    outliers['perc'] = round(outliers['count_outliers']/outliers['count_lines']*100, 2)
    outliers.sort_values(by='perc', ascending=False, inplace=True)

    return sns.barplot(data=outliers, x='perc', y=nivel)



def plot_inicial(df):

    # LISTAR COLUNAS DO DATAFRAME
    cols = df.columns.tolist()

    # PERCORRER COLUNAS
    for col in cols:

        # REDIMENSIONAR GRAFICO
        plt.subplots(figsize=(15,5))
        
        # SE MENOR QUE 8 CARDINALIDADE DOS DADOS CRIAR COUNTPLOT, SE NÃO, CRIAR HISTOGRAMA
        if len(df[col].unique()) < 8:
            sns.countplot(data=df, x=col)

        else:
            sns.histplot(data=df, x=col, kde=True)
        
        # SET TITLE + X E Y LABELS
        plt.title(col)
        plt.xlabel('')
        plt.ylabel('')
        plt.show()



def axs_show(axs, ax_titles=None):

    # PERCORRER AX'S E REMOVER LABELS
    for i, ax in enumerate(axs):
        ax.set_xlabel('')
        ax.set_ylabel('')

        # SE TIVER TITULO PLOTAR
        if ax_titles != None: ax.set_title(ax_titles[i])

    # AJUSTAR LAYOUT E ESPAÇAMENTO ENTRE FIGURAS
    plt.tight_layout()



def decomposicao(df, col, period, model='multiplicative', title=''):

    # DATA + COL
    df = df[['Data', col]]
    
    # SET INDEX
    df.set_index('Data', inplace=True)
    
    # PLOT
    result = seasonal_decompose(df, model=model, period=period)
    fig = result.plot()
    fig.suptitle(f'Decomposição Sazonal da Série Temporal {title}'.strip(), fontsize=16)



def maps(df, value):
    
    # MÉDIA DOS MUNICIPIOS
    df_fogo = df.groupby('Municipio', as_index=False).agg({'Latitude':'mean'
                                                          ,'Longitude':'mean'
                                                          ,value:'mean'})
    
    # ROUND RISCO DE FOGO
    df_fogo[value] = round(df_fogo[value], 2)

    # PLOT GEO
    fig = px.scatter_geo(df_fogo
                       ,lat='Latitude'
                       ,lon='Longitude'
                       ,color=value
                       ,color_continuous_scale='Reds'
                       ,hover_name='Municipio'
                       ,title=f'{value} - Média')
    
    # CENTRALIZAR + ZOOM MAPA - REGIÃO BRASIL
    fig.update_geos(
        center={"lat": np.mean(df['Latitude']), "lon": np.mean(df['Longitude'])},
        projection_scale=3.8)

    # PLOT
    fig.show()



def outliers_biomas(df, bioma, ax_titles):

    # AX PLOT
    _, ax = plt.subplots(1,2, figsize=(20,5))
    
    # BIOMA FILTRO
    df_temp = df[df['Bioma'] == bioma]
    
    # PLOT AX1 E AX2
    sns.lineplot(ax=ax[0], data=df_temp, y='DiaSemChuva', x='Data')
    sns.boxplot(ax=ax[1], data=df_temp, x='DiaSemChuva', y='Estado')
    
    # AJUSTAR PLOTS
    axs_show(ax, ax_titles=ax_titles)



def anomalias(df, filtro):

    # FILTRO
    df = df[df['UF_BIOMA'] == filtro]
    filtro = filtro.replace('_', ' ')

    # DIAS SEM CHUVA
    df['DiaSemChuva_outliers'] = np.where(df['outliers'] == 1, df['DiaSemChuva'], np.nan)

    # PRECIPITAÇÃO
    df['Precipitacao_outliers'] = np.where(df['outliers'] == 1, df['Precipitacao'], np.nan)

    # RISCO DE FOGO
    df['RiscoFogo_outliers'] = np.where(df['outliers'] == 1, df['RiscoFogo'], np.nan)


    # FIGSIZE
    plt.subplots(figsize=(25,5))

    # PLOT
    plt.plot(df['DiaSemChuva'], label='DiaSemChuva')
    plt.plot(df['DiaSemChuva_outliers'], 'ro', label='Outliers')
    plt.title(f'Dia Sem Chuva {filtro}')
    plt.legend()
    plt.show()


    # FIGSIZE
    plt.subplots(figsize=(25,5))

    # PLOT
    plt.plot(df['Precipitacao'], label='Precipitacao')
    plt.plot(df['Precipitacao_outliers'], 'ro', label='Outliers')
    plt.title(f'Precipitacao {filtro}')
    plt.legend()
    plt.show()


    # FIGSIZE
    plt.subplots(figsize=(25,5))

    # PLOT
    plt.plot(df['RiscoFogo'], label='RiscoFogo')
    plt.plot(df['RiscoFogo_outliers'], 'ro', label='Outliers')
    plt.title(f'Risco de Fogo {filtro}')
    plt.legend()
    plt.show()



def periodo(df, x, y, ax_title):

    _, ax = plt.subplots(2,1,figsize=(15,8))
    sns.lineplot(data=df, ax=ax[0], x=x, y=y)
    sns.boxplot(data=df, ax=ax[1], x=x, y=y)
    axs_show(ax, ax_titles=ax_title)
    plt.show()



def predicted(df, title):
    
    plt.subplots(figsize=(15,5))
    sns.lineplot(data=df, y='RiscoFogo', x='Data', label='Risco de Fogo')
    sns.lineplot(data=df[df['forecast'].notna()], y='forecast', x='Data', label='Forecast')
    plt.xlabel('')
    plt.ylabel('')
    plt.title(title)
    plt.show()