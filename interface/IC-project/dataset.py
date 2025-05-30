import pandas as pd
import numpy as np
import json
import sys
import os

# Diretório onde os arquivos estão localizados
diretorio = 'public/dataset/'

# Recebe o nome do arquivo via argumento
if len(sys.argv) > 1:
    arquivo = sys.argv[1]
else:
    raise Exception("Nome do arquivo não informado.")

# Caminho completo do arquivo
caminho_arquivo = os.path.join(diretorio, arquivo)

# Verifica se o arquivo é CSV ou XLSX e carrega
if arquivo.endswith('.csv'):
    df = pd.read_csv(caminho_arquivo)
elif arquivo.endswith('.xlsx'):
    df = pd.read_excel(caminho_arquivo)
else:
    raise Exception("Formato de arquivo não suportado.")

# Número de variáveis e lista de variáveis
output = {
    "num_variaveis": max(0, df.shape[1] - 1),
    "variaveis": df.columns.tolist()
}

# Total de registros
total = len(df)

# Inicia o DataFrame da descritiva
descritiva = pd.DataFrame(index=df.columns)

# Tipo de dado
descritiva['Type'] = df.dtypes.astype(str)

# Total de registros, nulos e % de nulos
descritiva['Total'] = total
descritiva['Nulls'] = df.isnull().sum()
descritiva['% of Nulls'] = (
    df.isnull().sum() / total * 100).round(2).astype(str) + '%'

# Valores únicos
descritiva['Unique Values'] = df.nunique()

# Valor mais frequente e sua frequência


def modo_unico(x):
    try:
        return x.mode().iloc[0]
    except:
        return '-'


descritiva['Most Frequent'] = df.apply(modo_unico)
descritiva['Frequency of the Most Freq.'] = df.apply(
    lambda x: x.value_counts().iloc[0] if not x.isnull().all() else '-'
)

# Estatísticas numéricas
descritiva['Average'] = df.select_dtypes(include=np.number).mean()
descritiva['Standard Deviation'] = df.select_dtypes(include=np.number).std()
descritiva['Min'] = df.select_dtypes(include=np.number).min()
descritiva['Q1 (25%)'] = df.select_dtypes(include=np.number).quantile(0.25)
descritiva['Q2 (50%)'] = df.select_dtypes(include=np.number).quantile(0.50)
descritiva['Q3 (75%)'] = df.select_dtypes(include=np.number).quantile(0.75)
descritiva['Max'] = df.select_dtypes(include=np.number).max()

# Preenche os NaNs restantes
descritiva = descritiva.fillna('-')

# Transforma para dicionário serializável em JSON
descritiva_str = descritiva.reset_index().rename(
    columns={'index': 'Feature'}).astype(str)
descritiva_json = descritiva_str.to_dict(orient='records')
output["descritiva_base"] = descritiva_json

# Salva o JSON no arquivo
os.makedirs('public/data_preprocess', exist_ok=True)
with open('public/data_preprocess/descritiva_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

# Escreve no stdout como JSON para ser capturado pelo app.js
sys.stdout.write(json.dumps(output, ensure_ascii=False))
