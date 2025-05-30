import pandas as pd
import os
import sys
import json
import numpy as np


# Diretório onde os arquivos CSV estão localizados
diretorio = 'public/dataset/'

# Lista todos os arquivos no diretório
arquivos = os.listdir(diretorio)

# Verifica se há algum arquivo CSV na lista
arquivos_csv = [arquivo for arquivo in arquivos if arquivo.endswith('.csv')]

# Se houver pelo menos um arquivo CSV, lê o primeiro arquivo encontrado
if arquivos_csv:
    arquivo_csv = arquivos_csv[0]
    caminho_arquivo = os.path.join(diretorio, arquivo_csv)

    # Realiza a leitura do arquivo CSV
    df = pd.read_csv(caminho_arquivo)

    # Total de registros
    total = len(df)

    # Inicia o DataFrame da descritiva
    descritiva = pd.DataFrame(index=df.columns)

    # Tipo de dado
    descritiva['Type'] = df.dtypes

    # Total de registros, nulos e % de nulos
    descritiva['Total'] = total
    descritiva['Nulls'] = df.isnull().sum()
    descritiva['% of Nulls'] = (
        df.isnull().sum() / total * 100).round(2).astype(str) + '%'

    # Valores únicos
    descritiva['Unique Values'] = df.nunique()

    def modo_unico(x):
        try:
            return x.mode().iloc[0]
        except:
            return '-'

    # Valor mais frequente e frequência
    descritiva['Most Frequent'] = df.mode().iloc[0]
    descritiva['Frequency of the Most Freq.'] = df.apply(
        lambda x: x.value_counts().iloc[0] if not x.isnull().all() else np.nan)

    # Estatísticas numéricas
    descritiva['Average'] = df.select_dtypes(include=np.number).mean()
    descritiva['Standard Deviation'] = df.select_dtypes(
        include=np.number).std()
    descritiva['Min'] = df.select_dtypes(include=np.number).min()
    descritiva['Q1 (25%)'] = df.select_dtypes(include=np.number).quantile(0.25)
    descritiva['Q2 (50% - Median)'] = df.select_dtypes(include=np.number).quantile(0.50)
    descritiva['Q3 (75%)'] = df.select_dtypes(include=np.number).quantile(0.75)
    descritiva['Max'] = df.select_dtypes(include=np.number).max()

    descritiva = descritiva.fillna('-')

    descritiva_str = descritiva.reset_index().rename(
        columns={'index': 'Feature'}).astype(str)
    descritiva_json = descritiva_str.to_dict(orient='records')

    sys.stdout.write(json.dumps(descritiva_json))
else:
    raise FileNotFoundError(
        "Nenhum arquivo CSV encontrado no diretório especificado.")
