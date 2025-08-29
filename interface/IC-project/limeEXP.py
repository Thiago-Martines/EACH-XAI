from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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
from modelUtils import ModelMetadata

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

# Identificar o tipo de modelo e criar o explicador LIME apropriado
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

# Acesse o scaler do pipeline
scaler = pipeline.named_steps['preprocessor'].transformers_[
    1][1]  # Obter o scaler do pipeline

# Criar explicador LIME
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_transformed,
    feature_names=selected_feature_names,
    class_names=class_names if 'encoder' not in locals() else encoder.classes_,
    mode='classification',
    discretize_continuous=True,
    categorical_features=[],
    random_state=None,
    verbose=False,
    sample_around_instance=True,
    training_data_stats=None
)


# Função para verificar consistência entre LIME e modelo
def check_lime_model_consistency(sample_transformed, model, explainer, tolerance=0.1):
    # Predição do modelo original
    sample_for_prediction = sample_transformed.reshape(1, -1)
    model_prediction = model.predict(sample_for_prediction)[0]
    model_probabilities = model.predict_proba(sample_for_prediction)[0]

    # Gerar explicação LIME
    lime_explanation = explainer.explain_instance(
        sample_transformed,
        model.predict_proba,
        num_features=len(selected_feature_names),
        num_samples=5000,
        labels=list(range(n_classes)) if not is_binary else [1],
        distance_metric='euclidean',
        model_regressor=None
    )

    # Obter predição local do LIME
    lime_local_pred = lime_explanation.local_pred

    if is_binary:
        # Para classificação binária, comparar a classe predita
        lime_predicted_class = 1 if lime_local_pred[0] > 0.5 else 0
        is_consistent = (lime_predicted_class == model_prediction)

        # Verificar também se as probabilidades são próximas
        prob_diff = abs(lime_local_pred[0] - model_probabilities[1])
        is_prob_consistent = prob_diff <= tolerance

        is_consistent = is_consistent and is_prob_consistent

    else:
        # Para classificação multiclasse, verificar a classe com maior probabilidade
        lime_predicted_class = np.argmax(lime_local_pred)
        is_consistent = (lime_predicted_class == model_prediction)

        # Verificar se a probabilidade da classe predita é próxima
        prob_diff = abs(
            lime_local_pred[model_prediction] - model_probabilities[model_prediction])
        is_prob_consistent = prob_diff <= tolerance

        is_consistent = is_consistent and is_prob_consistent

    return is_consistent, lime_explanation, model_prediction, lime_local_pred


# Função para reverter a normalização
def inverse_transform_feature(scaler, value, feature_name, remaining_columns):
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


def create_lime_explanation_single(lime_explanation, selected_feature_names, sample_transformed_single, scaler, remaining_columns, is_binary, n_classes, class_names):
    explanation_dict = {}
    sample_values = sample_transformed_single

    # Função aprimorada para extrair o nome da feature das descrições do LIME
    def extract_feature_name(feature_desc):
        import re

        # Lista de possíveis padrões do LIME
        patterns = [
            r'(\w+)\s*[<>=]',  # feature_name <= value ou feature_name > value
            r'[\d\.-]+\s*[<>=]\s*(\w+)',  # value <= feature_name
            r'(\w+)\s+[<>=]',  # feature_name < value (com espaço)
            r'^(\w+)',  # primeira palavra como fallback
        ]

        for pattern in patterns:
            match = re.search(pattern, feature_desc)
            if match:
                potential_name = match.group(1)
                # Verificar se o nome extraído está na lista de features
                if potential_name in selected_feature_names:
                    return potential_name

        # Fallback final: tentar encontrar qualquer feature name na descrição
        for feature_name in selected_feature_names:
            if feature_name in feature_desc:
                return feature_name

        return feature_desc.split(' ')[0]

    # Função para converter valores normalizados de volta aos valores originais na descrição
    def convert_description_to_original_values(feature_desc, feature_name):
        import re

        # Encontrar todos os valores numéricos na descrição
        values = re.findall(r'[\d\.-]+', feature_desc)

        if not values:
            return feature_desc

        # Converter cada valor encontrado
        converted_desc = feature_desc
        for value_str in values:
            try:
                normalized_value = float(value_str)
                original_value = inverse_transform_feature(
                    scaler, normalized_value, feature_name, remaining_columns)
                # Substituir o valor normalizado pelo original na descrição
                converted_desc = converted_desc.replace(
                    value_str, f"{original_value:.4f}", 1)
            except:
                continue

        return converted_desc

    # Mapeia as explicações fornecidas pelo LIME
    def parse_lime_explanation(lime_list):
        parsed = {}
        for feature_desc, lime_value in lime_list:
            feature_name = extract_feature_name(feature_desc)
            # Converter a descrição para valores originais
            original_desc = convert_description_to_original_values(
                feature_desc, feature_name)
            parsed[feature_name] = {
                'lime_value': lime_value,
                'lime_description': original_desc
            }
        return parsed

    if is_binary:
        lime_features = parse_lime_explanation(lime_explanation.as_list())

        # Criar estrutura com valor original da feature como chave principal
        for feature_name in selected_feature_names:
            feature_idx = selected_feature_names.index(feature_name)
            normalized_value = sample_values[feature_idx]

            try:
                original_value = inverse_transform_feature(
                    scaler, normalized_value, feature_name, remaining_columns)
            except Exception:
                original_value = normalized_value

            feature_key = f"{feature_name} = {original_value:.4f}"

            if feature_name in lime_features:
                explanation_dict[feature_key] = {
                    'lime_value': float(lime_features[feature_name]['lime_value']),
                    'lime_description': lime_features[feature_name]['lime_description']
                }
            else:
                explanation_dict[feature_key] = {
                    'lime_value': "not considered",
                    'lime_description': "feature not used in explanation"
                }

    else:
        for class_idx, class_name in enumerate(class_names):
            try:
                class_lime_features = parse_lime_explanation(
                    lime_explanation.as_list(label=class_idx))
                explanation_dict[f"Classe_{class_name}"] = {}

                # Criar estrutura com valor original da feature como chave principal
                for feature_name in selected_feature_names:
                    feature_idx = selected_feature_names.index(feature_name)
                    normalized_value = sample_values[feature_idx]

                    try:
                        original_value = inverse_transform_feature(
                            scaler, normalized_value, feature_name, remaining_columns)
                    except Exception:
                        original_value = normalized_value

                    feature_key = f"{feature_name} = {original_value:.4f}"

                    if feature_name in class_lime_features:
                        explanation_dict[f"Classe_{class_name}"][feature_key] = {
                            'lime_value': float(class_lime_features[feature_name]['lime_value']),
                            'lime_description': class_lime_features[feature_name]['lime_description']
                        }
                    else:
                        explanation_dict[f"Classe_{class_name}"][feature_key] = {
                            'lime_value': "not considered",
                            'lime_description': "feature not used in explanation"
                        }

            except Exception as e:
                print(f"Erro na classe {class_name}: {e}", file=sys.stderr)
                continue

    return explanation_dict


# Selecionar amostras com predições consistentes
consistent_samples = []
max_attempts = min(1000, X_test_transformed.shape[0])
attempts = 0
np.random.seed()

# Gerar índices aleatórios únicos para testar
available_indices = list(range(X_test_transformed.shape[0]))
np.random.shuffle(available_indices)

for sample_idx in available_indices:
    if len(consistent_samples) >= 10:
        break

    if attempts >= max_attempts:
        break

    attempts += 1
    sample_transformed = X_test_transformed[sample_idx]

    try:
        is_consistent, lime_exp, model_pred, lime_pred = check_lime_model_consistency(
            sample_transformed, model, explainer, tolerance=0.15
        )

        if is_consistent:
            consistent_samples.append({
                'original_index': sample_idx,
                'sample_transformed': sample_transformed,
                'lime_explanation': lime_exp,
                'model_prediction': model_pred,
                'lime_prediction': lime_pred
            })

    except Exception as e:
        print(f"Erro ao processar amostra {sample_idx}: {e}", file=sys.stderr)
        continue

if len(consistent_samples) == 0:
    # Tentar novamente com tolerância maior
    for sample_idx in available_indices[:50]:
        sample_transformed = X_test_transformed[sample_idx]

        try:
            is_consistent, lime_exp, model_pred, lime_pred = check_lime_model_consistency(
                sample_transformed, model, explainer, tolerance=0.3
            )

            if is_consistent:
                consistent_samples.append({
                    'original_index': sample_idx,
                    'sample_transformed': sample_transformed,
                    'lime_explanation': lime_exp,
                    'model_prediction': model_pred,
                    'lime_prediction': lime_pred
                })

                if len(consistent_samples) >= 10:
                    break

        except Exception as e:
            continue

# Processar amostras consistentes encontradas
all_explanations = {}
all_predictions_info = {}

for idx, sample_data in enumerate(consistent_samples):
    sample_idx = sample_data['original_index']
    sample_transformed = sample_data['sample_transformed']
    lime_explanation = sample_data['lime_explanation']
    model_prediction = sample_data['model_prediction']
    lime_prediction = sample_data['lime_prediction']

    # Gerar a explicação LIME para esta amostra
    lime_explanation_dict = create_lime_explanation_single(
        lime_explanation, selected_feature_names, sample_transformed,
        scaler, remaining_columns, is_binary, n_classes, class_names)

    # Ordenar por importância absoluta
    if is_binary:
        # Para classificação binária, ordenar por importância absoluta do lime_value
        sorted_explanation = dict(sorted(
            lime_explanation_dict.items(),
            key=lambda x: abs(x[1]['lime_value']) if isinstance(
                x[1], dict) and isinstance(x[1].get('lime_value'), (int, float)) else 0,
            reverse=True
        ))
    else:
        # Para classificação multiclasse, ordenar cada classe
        sorted_explanation = {}
        for class_key, class_explanations in lime_explanation_dict.items():
            if isinstance(class_explanations, dict) and class_explanations:
                sorted_class_explanations = dict(sorted(
                    class_explanations.items(),
                    key=lambda x: abs(x[1]['lime_value']) if isinstance(
                        x[1], dict) and isinstance(x[1].get('lime_value'), (int, float)) else 0,
                    reverse=True
                ))
                sorted_explanation[class_key] = sorted_class_explanations

    # Armazenar a explicação usando numeração sequencial
    all_explanations[f"sample_{idx + 1}"] = sorted_explanation

    # Obter informações da predição
    sample_for_prediction = sample_transformed.reshape(1, -1)
    prediction = model.predict(sample_for_prediction)[0]
    probabilities = model.predict_proba(sample_for_prediction)[0]

    prediction_info = {
        "sample_index": int(sample_idx),
        "prediction": int(prediction),
        "lime_consistency": "consistent",
        "lime_local_prediction": lime_prediction.tolist() if hasattr(lime_prediction, 'tolist') else lime_prediction
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
        "n_samples_analyzed": len(consistent_samples),
        "consistency_check": "enabled",
        "selected_features": selected_feature_names,
        "classification_report": class_report
    },
    "predictions": all_predictions_info,
    "lime_explanations": all_explanations
}


# Imprimir APENAS o JSON para stdout
print(json.dumps(final_results, indent=2, ensure_ascii=False))
