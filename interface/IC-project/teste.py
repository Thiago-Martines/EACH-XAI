import joblib
import numpy as np
import lime
import lime.lime_tabular
import os
import pandas as pd
import re
import sys
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
import matplotlib.pyplot as plt

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
    df = pd.read_csv(caminho_arquivo).dropna()

else:
    raise FileNotFoundError(
        "Nenhum arquivo CSV encontrado no diretório especificado.")


# Função para carregar o arquivo e definir as variáveis
def load_grid_config(file_path):
    global target_columns

    target_columns = []

    # Regex
    patterns = {
        "Var Target": r"Var Target:\s*([\w, ]*)(?=\nVar Deleted|$)"
    }

    with open(file_path, 'r') as file:
        content = file.read()

        # Captura da variável alvo
        target_columns_values = re.search(
            patterns["Var Target"], content)
        if target_columns_values:
            target_columns = [target_col.strip()
                              for target_col in target_columns_values.group(1).split(",")]
        target_columns = [
            target_col for target_col in target_columns if target_col != '']


# Caminho do arquivo de configuração
file_path = './temp_config.txt'
# Carregar as variáveis do arquivo
load_grid_config(file_path)


sample_path = './public/sample/sample.csv'
sample = pd.read_csv(sample_path)
sample = sample.drop(columns=[target_columns[0]])


# Divisão do dataset
X_df = df.drop(columns=[target_columns[0]])
y_df = df[target_columns[0]]
class_names = y_df.unique()

X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_df, test_size=0.2, random_state=42)

if y_df.dtype == 'object' or y_df.dtype.name == 'category':
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

pipeline = joblib.load("public/model/model.pkl")

deleted_columns = next(
    t[2] for t in pipeline.named_steps['preprocessor'].transformers if t[1] == 'drop')


# Aplicar o pré-processamento e o imputer aos dados
X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
X_train_transformed = pipeline.named_steps['imputer'].transform(
    X_train_transformed)

sample_transformed = pipeline.named_steps['preprocessor'].transform(sample)
sample_transformed = pipeline.named_steps['imputer'].transform(
    sample_transformed)

transformed_columns = pipeline.named_steps['preprocessor'].get_feature_names_out(
)


# Obter os nomes das colunas restantes após o drop
remaining_columns = [
    col for col in X_train.columns if col not in deleted_columns]


# Obter os nomes das features após o SelectKBest
kbest = pipeline.named_steps['kbest']


# Índices das features selecionadas
selected_feature_indices = kbest.get_support()
selected_feature_names = [remaining_columns[idx] for idx in range(
    len(remaining_columns)) if selected_feature_indices[idx]]


# Aplicar o SelectKBest aos dados transformados
X_train_transformed = pipeline.named_steps['kbest'].transform(
    X_train_transformed)
sample_transformed = pipeline.named_steps['kbest'].transform(
    sample_transformed)


# Criar o explicador LIME com os dados transformados e as features selecionadas
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_transformed,
    feature_names=selected_feature_names,
    class_names=class_names,
    mode='classification',
    verbose=True,
    random_state=42
)


# Gerar explicação para a amostra
explanation = explainer.explain_instance(
    data_row=sample_transformed[0],
    predict_fn=pipeline.named_steps['mlp'].predict_proba,
    num_features=len(selected_feature_names)
)


# Acesse o scaler e os parâmetros necessários
scaler = pipeline.named_steps['preprocessor'].transformers_[
    1][1]  # Obter o scaler do pipeline


# Função para reverter a normalização
def inverse_transform(scaler, value, feature_index):
    if isinstance(scaler, MinMaxScaler):
        X_min = scaler.data_min_[feature_index]
        X_max = scaler.data_max_[feature_index]
        return value * (X_max - X_min) + X_min
    elif isinstance(scaler, StandardScaler):
        mean = scaler.mean_[feature_index]
        std = scaler.scale_[feature_index]
        return value * std + mean
    elif isinstance(scaler, RobustScaler):
        center = scaler.center_[feature_index]
        scale = scaler.scale_[feature_index]
        return value * scale + center
    else:
        raise ValueError(
            "Scaler não suportado. Use MinMaxScaler, StandardScaler ou RobustScaler.")


def desnormalize_lime_explanation(explanation, scaler, selected_feature_names):
    desnormalized_explanation = []
    for feature, weight in explanation.as_list():
        # Verificar se a feature está em um intervalo (ex: -0.35 < pedigree_func <= 0.00)
        if '<' in feature and '<=' in feature and feature.count('<') == 2:
            # Extrair o nome da feature e os valores do intervalo
            parts = feature.split()
            feature_name = parts[2]  # O nome da feature está no meio
            lower_bound = float(parts[0])  # Limite inferior
            upper_bound = float(parts[4])  # Limite superior
            feature_index = selected_feature_names.index(feature_name)

            # Reverter a normalização para os limites do intervalo
            original_lower = inverse_transform(
                scaler, lower_bound, feature_index)
            original_upper = inverse_transform(
                scaler, upper_bound, feature_index)

            # Criar a nova condição com os valores originais
            desnormalized_feature = f"{original_lower} < {feature_name} <= {original_upper}"

        # Verificar se a feature tem uma condição simples (ex: plasma_glucose_conc > 0.56)
        elif '<=' in feature or '>' in feature:
            parts = feature.split()
            feature_name = parts[0]  # O nome da feature está no início
            condition = parts[1]  # A condição (<= ou >)
            value = float(parts[2])  # O valor normalizado
            feature_index = selected_feature_names.index(feature_name)

            # Reverter a normalização
            original_value = inverse_transform(scaler, value, feature_index)

            # Criar a nova condição com o valor original
            desnormalized_feature = f"{feature_name} {condition} {original_value}"

        else:
            raise ValueError(f"Formato de condição não suportado: {feature}")

        # Adicionar a feature desnormalizada e o peso à lista
        desnormalized_explanation.append((desnormalized_feature, weight))

    return desnormalized_explanation


# Aplicar a desnormalização à explicação LIME
desnormalized_explanation = desnormalize_lime_explanation(
    explanation, scaler, selected_feature_names)


# Convertendo para um dicionário
explanation_dict = {feature: weight for feature,
                    weight in desnormalized_explanation}


print(json.dumps(explanation_dict))
