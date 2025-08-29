from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import joblib
import numpy as np
import shap
import os
import pandas as pd
import re
import sys
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
import matplotlib.pyplot as plt
from modelUtils import ModelMetadata

# Diretório onde os arquivos CSV estão localizados
diretório = 'public/dataset/'

# Lista todos os arquivos no diretório
arquivos = os.listdir(diretório)

# Verifica se há algum arquivo CSV na lista
arquivos_csv = [arquivo for arquivo in arquivos if arquivo.endswith('.csv')]

# Se houver pelo menos um arquivo CSV, lê o primeiro arquivo encontrado
if arquivos_csv:
    arquivo_csv = arquivos_csv[0]
    caminho_arquivo = os.path.join(diretório, arquivo_csv)

    # Realiza a leitura do arquivo CSV
    df = pd.read_csv(caminho_arquivo).dropna()

else:
    raise FileNotFoundError(
        "Nenhum arquivo CSV encontrado no diretório especificado.")

# Carregar o modelo com metadados
model_with_metadata = joblib.load("public/model/model.pkl")

# Acessar os metadados
target_column = model_with_metadata.target_column
class_report = model_with_metadata.classification_report

# Acessar o pipeline
pipeline = model_with_metadata.pipeline

# Divisão do dataset
X_df = df.drop(columns=[target_column[0]])
y_df = df[target_column[0]]

X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_df, test_size=0.2, random_state=42)

class_names = y_df.unique()

# Determinar se é classificação binária ou multiclasse
is_binary = len(class_names) == 2
n_classes = len(class_names)

if y_df.dtype == 'object' or y_df.dtype.name == 'category':
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

deleted_columns = next(
    t[2] for t in pipeline.named_steps['preprocessor'].transformers if t[1] == 'drop')

# Aplicar o pré-processamento e o imputer aos dados
X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
X_train_transformed = pipeline.named_steps['imputer'].transform(
    X_train_transformed)

X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)
X_test_transformed = pipeline.named_steps['imputer'].transform(
    X_test_transformed)

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
X_test_transformed = pipeline.named_steps['kbest'].transform(
    X_test_transformed)

# Determinar o número de amostras a serem analisadas (máximo 10)
n_samples = min(10, X_test_transformed.shape[0])

# Selecionar amostras aleatórias e distintas do conjunto de teste
np.random.seed()
random_indices = np.random.choice(
    X_test_transformed.shape[0], size=n_samples, replace=False)
samples_transformed = X_test_transformed[random_indices]

# Identificar o tipo de modelo e criar o explicador SHAP apropriado
model_step_name = None
for step_name in pipeline.named_steps:
    if step_name in ['mlp', 'randomforest', 'rf', 'classifier']:
        model_step_name = step_name
        break

if model_step_name is None:
    # Fallback: procurar por qualquer estimador que não seja preprocessor, imputer ou kbest
    for step_name in pipeline.named_steps:
        if step_name not in ['preprocessor', 'imputer', 'kbest']:
            model_step_name = step_name
            break

model = pipeline.named_steps[model_step_name]

# Criar background data para o explicador
background_size = min(100, X_train_transformed.shape[0])
background_indices = np.random.choice(
    X_train_transformed.shape[0], background_size, replace=False)
background_data = X_train_transformed[background_indices]

# Verificar o tipo de modelo e criar o explicador apropriado
if isinstance(model, RandomForestClassifier):
    explainer = shap.TreeExplainer(model)
    # Para TreeExplainer, o expected_value vem do explicador
    expected_value = explainer.expected_value
elif isinstance(model, MLPClassifier):
    explainer = shap.KernelExplainer(model.predict_proba, background_data)
    # Para KernelExplainer, calcular o expected_value manualmente
    expected_value = model.predict_proba(background_data).mean(axis=0)

# Calcular E[f(x)] baseline
if isinstance(expected_value, np.ndarray):
    if is_binary:
        baseline_value = float(expected_value[1] if len(
            expected_value) > 1 else expected_value[0])
    else:
        baseline_value = expected_value.tolist()
else:
    baseline_value = float(expected_value)

# Acesse o scaler do pipeline
scaler = pipeline.named_steps['preprocessor'].transformers_[
    1][1]


# Função para reverter a normalização
def inverse_transform_feature(scaler, value, feature_name, remaining_columns):
    # Encontrar o índice da feature nas colunas restantes
    feature_index_in_remaining = remaining_columns.index(feature_name)

    if isinstance(scaler, MinMaxScaler):
        X_min = scaler.data_min_[feature_index_in_remaining]
        X_max = scaler.data_max_[feature_index_in_remaining]
        return value * (X_max - X_min) + X_min
    elif isinstance(scaler, StandardScaler):
        mean = scaler.mean_[feature_index_in_remaining]
        std = scaler.scale_[feature_index_in_remaining]
        return value * std + mean
    elif isinstance(scaler, RobustScaler):
        center = scaler.center_[feature_index_in_remaining]
        scale = scaler.scale_[feature_index_in_remaining]
        return value * scale + center
    else:
        raise ValueError(
            "Scaler não suportado. Use MinMaxScaler, StandardScaler ou RobustScaler.")


def create_shap_explanation_single(shap_values_single, selected_feature_names, sample_transformed_single, scaler, remaining_columns, is_binary, n_classes, class_names):
    explanation_dict = {}
    sample_values = sample_transformed_single

    if is_binary:
        # Para classificação binária
        for i, (feature_name, shap_value) in enumerate(zip(selected_feature_names, shap_values_single)):
            normalized_value = sample_values[i]

            # Reverter a normalização para obter o valor original
            try:
                original_value = inverse_transform_feature(
                    scaler, normalized_value, feature_name, remaining_columns)
            except Exception as e:
                original_value = normalized_value  # Usar valor normalizado como fallback

            # Criar uma descrição da feature com seu valor
            feature_description = f"{feature_name} = {original_value:.4f}"

            # Adicionar ao dicionário de explicação
            explanation_dict[feature_description] = float(shap_value)
    else:
        # Para classificação multiclasse,  criar explicação para cada classe
        for class_idx in range(n_classes):
            class_name = class_names[class_idx]
            explanation_dict[f"Classe_{class_name}"] = {}

            for i, feature_name in enumerate(selected_feature_names):
                # Obter o valor normalizado da feature
                normalized_value = sample_values[i]

                # Reverter a normalização para obter o valor original
                try:
                    original_value = inverse_transform_feature(
                        scaler, normalized_value, feature_name, remaining_columns)
                except Exception as e:
                    original_value = normalized_value

                feature_description = f"{feature_name} = {original_value:.4f}"

                # Obter o valor SHAP para esta feature e classe
                shap_value = shap_values_single[i, class_idx]

                # Adicionar ao dicionário de explicação
                explanation_dict[f"Classe_{class_name}"][feature_description] = float(
                    shap_value)

    return explanation_dict


# Processar todas as amostras
all_explanations = {}
all_predictions_info = {}

for idx, (sample_idx, sample_transformed) in enumerate(zip(random_indices, samples_transformed)):

    # Gerar explicação SHAP para a amostra atual
    sample_for_shap = sample_transformed.reshape(1, -1)
    shap_values = explainer(sample_for_shap)

    # Extrair valores SHAP para uma única amostra
    if isinstance(explainer, shap.TreeExplainer):
        if is_binary:
            if isinstance(shap_values.values, np.ndarray) and len(shap_values.values.shape) == 3:
                shap_values_single = shap_values.values[0, :, 1]
            elif hasattr(shap_values, 'values') and shap_values.values.ndim == 2:
                shap_values_single = shap_values.values[:, 1]
            else:
                shap_values_single = shap_values.values[0] if hasattr(
                    shap_values, 'values') else shap_values[0]
        else:
            if isinstance(shap_values.values, np.ndarray) and len(shap_values.values.shape) == 3:
                shap_values_single = shap_values.values[0]
            else:
                shap_values_single = shap_values.values[0] if hasattr(
                    shap_values, 'values') else shap_values[0]
    else:
        # Para KernelExplainer
        if len(shap_values.values.shape) == 3:
            if is_binary:
                shap_values_single = shap_values.values[0, :, 1]
            else:
                shap_values_single = shap_values.values[0]
        else:
            shap_values_single = shap_values.values[0]

    # Gerar a explicação SHAP para esta amostra
    shap_explanation = create_shap_explanation_single(
        shap_values_single, selected_feature_names, sample_transformed,
        scaler, remaining_columns, is_binary, n_classes, class_names)

    # Ordenar por importância
    if is_binary:
        sorted_explanation = dict(sorted(shap_explanation.items(),
                                         key=lambda x: abs(x[1]), reverse=True))
    else:
        sorted_explanation = {}
        for class_key, class_explanations in shap_explanation.items():
            sorted_explanation[class_key] = dict(sorted(class_explanations.items(),
                                                        key=lambda x: abs(x[1]), reverse=True))

    # Armazenar a explicação usando numeração sequencial
    all_explanations[f"sample_{idx + 1}"] = sorted_explanation

    # Obter informações da predição
    prediction = model.predict(sample_for_shap)[0]
    probabilities = model.predict_proba(sample_for_shap)[0]

    prediction_info = {
        "sample_index": int(sample_idx),
        "prediction": int(prediction)
    }

    # Adicionar nome da classe predita se encoder estiver disponível
    if 'encoder' in locals():
        predicted_class_name = encoder.inverse_transform([prediction])[0]
        prediction_info["predicted_class_name"] = predicted_class_name

        # Adicionar probabilidades com nomes das classes
        prediction_info["class_probabilities"] = {}
        encoder_classes = encoder.classes_
        for i, (class_name, prob) in enumerate(zip(encoder_classes, probabilities)):
            prediction_info["class_probabilities"][str(
                class_name)] = float(prob)
    else:
        # Usar nomes das classes originais
        prediction_info["class_probabilities"] = {}
        for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
            prediction_info["class_probabilities"][str(
                class_name)] = float(prob)

    all_predictions_info[f"sample_{idx + 1}"] = prediction_info

# Criar estrutura final dos resultados
final_results = {
    "metadata": {
        "target_column": target_column[0],
        "classification_type": "binary" if is_binary else "multiclass",
        "n_classes": n_classes,
        "class_names": class_names.tolist(),
        "n_samples_analyzed": n_samples,
        "selected_features": selected_feature_names,
        "baseline_value": baseline_value,
        "classification_report": class_report
    },
    "predictions": all_predictions_info,
    "shap_explanations": all_explanations
}

# Imprimir APENAS o JSON para stdout
print(json.dumps(final_results, indent=2, ensure_ascii=False))
