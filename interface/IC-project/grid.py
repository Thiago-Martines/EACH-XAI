import os
import pandas as pd
import optuna
import re
from io import StringIO
import sys
import json
import joblib
import numpy as np
import warnings
import time
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from modelUtils import ModelMetadata

optuna.logging.set_verbosity(optuna.logging.ERROR)

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
    global scalers, num_folds, min_k, max_k, score_functions, activation_functions, solvers, learning_rates, num_trials
    global max_learning_rate_init, min_learning_rate_init, min_alfa, max_alfa, hidden_layer_size, max_epochs, min_epochs
    global target_column, deleted_columns, score_metric, model_selection
    # Random Forest specific variables
    global min_n_estimators, max_n_estimators, min_max_depth, max_max_depth, max_features_options
    global criterion_options, min_min_samples_leaf, max_min_samples_leaf, min_min_samples_split, max_min_samples_split

    num_folds = None
    min_k = None
    max_k = None
    score_functions = []
    activation_functions = []
    solvers = []
    learning_rates = []
    max_learning_rate_init = None
    min_learning_rate_init = None
    min_alfa = None
    max_alfa = None
    hidden_layer_size = []
    max_epochs = None
    min_epochs = None
    num_trials = None
    target_column = []
    deleted_columns = []
    scalers = []
    score_metric = []
    model_selection = None

    # Random Forest variables
    min_n_estimators = None
    max_n_estimators = None
    min_max_depth = None
    max_max_depth = None
    max_features_options = []
    criterion_options = []
    min_min_samples_leaf = None
    max_min_samples_leaf = None
    min_min_samples_split = None
    max_min_samples_split = None

    # Regex
    patterns = {
        # Common patterns
        "NumTrials": r"NumTrials:\s*(\d+)",
        "Score Metric": r"Score Metric:\s*([\w, ]+)",
        "Var Target": r"Var Target:\s*([\w, ]*)(?=\nVar Deleted|$)",
        "Var Deleted": r"Var Deleted:\s*([\w, ]*)(?=\nScalers|$|[\n\r])",
        "Scalers": r"Scalers:\s*([\w, ]+)",
        "NumFolds": r"NumFolds:\s*(\d+)",
        "minK": r"minK:\s*(\d+)",
        "maxK": r"maxK:\s*(\d+)",
        "Score Function": r"Score Function:\s*([\w, ]+)",
        "Model Selection": r"Model Selection:\s*([\w ]+)",
        # MLP patterns
        "Activation Function": r"Activation Function:\s*([\w, ]+)",
        "Solver": r"Solver:\s*([\w, ]+)",
        "Learning Rate": r"Learning Rate:\s*([\w, ]+)",
        "maxLearningRateInit": r"maxLearningRateInit:\s*([\d.]+)",
        "minLearningRateInit": r"minLearningRateInit:\s*([\d.]+)",
        "minAlfa": r"minAlfa:\s*([\d.]+)",
        "maxAlfa": r"maxAlfa:\s*([\d.]+)",
        "Hidden Layer Sizes": r"Hidden Layer Sizes:\s*([\(\d+(\s*,\s*\d+)*\)]+)",
        "maxEpochs": r"maxEpochs:\s*(\d+)",
        "minEpochs": r"minEpochs:\s*(\d+)",
        # Random Forest patterns
        "minNEstimators": r"minNEstimators:\s*(\d+)",
        "maxNEstimators": r"maxNEstimators:\s*(\d+)",
        "minMaxDepth": r"minMaxDepth:\s*(\d+)",
        "maxMaxDepth": r"maxMaxDepth:\s*(\d+)",
        "Max Features": r"Max Features:\s*([\w, ]+)",
        "Criterion": r"Criterion:\s*([\w, ]+)",
        "minMinSamplesLeaf": r"minMinSamplesLeaf:\s*(\d+)",
        "maxMinSamplesLeaf": r"maxMinSamplesLeaf:\s*(\d+)",
        "minMinSamplesSplit": r"minMinSamplesSplit:\s*(\d+)",
        "maxMinSamplesSplit": r"maxMinSamplesSplit:\s*(\d+)",
    }

    with open(file_path, 'r') as file:
        content = file.read()

        # Usando regex para capturar os valores
        # Número de folds
        num_folds = int(re.search(patterns["NumFolds"], content).group(
            1)) if re.search(patterns["NumFolds"], content) else None

        # Número de Trials
        num_trials = int(re.search(patterns["NumTrials"], content).group(
            1)) if re.search(patterns["NumTrials"], content) else None

        # Métrica de score
        score_metric_match = re.search(patterns["Score Metric"], content)
        score_metric = score_metric_match.group(
            1).strip() if score_metric_match else None

        # Model Selection
        model_selection_match = re.search(patterns["Model Selection"], content)
        model_selection = model_selection_match.group(
            1).strip() if model_selection_match else None

        # Função de Score Function
        score_function_values = re.search(patterns["Score Function"], content)
        if score_function_values:
            score_functions = [
                func.strip() for func in score_function_values.group(1).split(",")]

        # Escalonadores
        scaler_values = re.search(patterns["Scalers"], content)
        if scaler_values:
            scalers = [scaler.strip()
                       for scaler in scaler_values.group(1).split(",")]

        # Valores K
        min_k = int(re.search(patterns["minK"], content).group(
            1)) if re.search(patterns["minK"], content) else None
        max_k = int(re.search(patterns["maxK"], content).group(
            1)) if re.search(patterns["maxK"], content) else None

        # Captura da variável alvo e as excluídas
        target_column_values = re.search(patterns["Var Target"], content)
        if target_column_values:
            target_column = [target_col.strip()
                             for target_col in target_column_values.group(1).split(",")]
        target_column = [
            target_col for target_col in target_column if target_col != '']

        deleted_columns_values = re.search(patterns["Var Deleted"], content)
        if deleted_columns_values:
            deleted_columns = [
                deleted_col.strip() for deleted_col in deleted_columns_values.group(1).split(",")]
        deleted_columns = [
            deleted_col for deleted_col in deleted_columns if deleted_col != '']

        # MLP specific parameters
        if model_selection == "MLP Classifier":
            # Funções de Activation Function
            activation_function_values = re.search(
                patterns["Activation Function"], content)
            if activation_function_values:
                activation_functions = [
                    func.strip() for func in activation_function_values.group(1).split(",")]

            # Solvers
            solver_values = re.search(patterns["Solver"], content)
            if solver_values:
                solvers = [solver.strip()
                           for solver in solver_values.group(1).split(",")]

            # Learning Rates
            learning_rate_values = re.search(
                patterns["Learning Rate"], content)
            if learning_rate_values:
                learning_rates = [rate.strip()
                                  for rate in learning_rate_values.group(1).split(",")]

            # Valores numéricos
            max_learning_rate_init = float(re.search(patterns["maxLearningRateInit"], content).group(
                1)) if re.search(patterns["maxLearningRateInit"], content) else None
            min_learning_rate_init = float(re.search(patterns["minLearningRateInit"], content).group(
                1)) if re.search(patterns["minLearningRateInit"], content) else None

            min_alfa = float(re.search(patterns["minAlfa"], content).group(
                1)) if re.search(patterns["minAlfa"], content) else None
            max_alfa = float(re.search(patterns["maxAlfa"], content).group(
                1)) if re.search(patterns["maxAlfa"], content) else None

            min_epochs = int(re.search(patterns["minEpochs"], content).group(
                1)) if re.search(patterns["minEpochs"], content) else None
            max_epochs = int(re.search(patterns["maxEpochs"], content).group(
                1)) if re.search(patterns["maxEpochs"], content) else None

            # Hidden Layer Sizes
            hidden_layer_values = re.search(
                patterns["Hidden Layer Sizes"], content)
            if hidden_layer_values:
                for size in hidden_layer_values.group(1).split("),"):
                    # Remover espaços extras e parênteses
                    clean_size = re.sub(r'[()]', '', size).strip()
                    # Se houver mais de um valor, armazene como tupla
                    if ',' in clean_size:
                        hidden_layer_size.append(
                            tuple(map(int, clean_size.split(','))))
                    else:
                        # Como tupla com um único valor
                        hidden_layer_size.append((int(clean_size),))

        # Random Forest specific parameters
        elif model_selection == "Random Forest":
            min_n_estimators = int(re.search(patterns["minNEstimators"], content).group(
                1)) if re.search(patterns["minNEstimators"], content) else None
            max_n_estimators = int(re.search(patterns["maxNEstimators"], content).group(
                1)) if re.search(patterns["maxNEstimators"], content) else None

            min_max_depth = int(re.search(patterns["minMaxDepth"], content).group(
                1)) if re.search(patterns["minMaxDepth"], content) else None
            max_max_depth = int(re.search(patterns["maxMaxDepth"], content).group(
                1)) if re.search(patterns["maxMaxDepth"], content) else None

            min_min_samples_leaf = int(re.search(patterns["minMinSamplesLeaf"], content).group(
                1)) if re.search(patterns["minMinSamplesLeaf"], content) else None
            max_min_samples_leaf = int(re.search(patterns["maxMinSamplesLeaf"], content).group(
                1)) if re.search(patterns["maxMinSamplesLeaf"], content) else None

            min_min_samples_split = int(re.search(patterns["minMinSamplesSplit"], content).group(
                1)) if re.search(patterns["minMinSamplesSplit"], content) else None
            max_min_samples_split = int(re.search(patterns["maxMinSamplesSplit"], content).group(
                1)) if re.search(patterns["maxMinSamplesSplit"], content) else None

            # Max Features
            max_features_values = re.search(patterns["Max Features"], content)
            if max_features_values:
                max_features_options = [
                    feat.strip() for feat in max_features_values.group(1).split(",")]
                # Convert 'None' string to None object
                max_features_options = [
                    None if feat == 'None' else feat for feat in max_features_options]

            # Criterion
            criterion_values = re.search(patterns["Criterion"], content)
            if criterion_values:
                criterion_options = [
                    crit.strip() for crit in criterion_values.group(1).split(",")]


# Caminho do arquivo de configuração
# file_path = sys.argv[1]
file_path = "./temp_config.txt"

# Carregar as variáveis do arquivo
load_grid_config(file_path)

# Divisão do dataset
X_df = df.drop(columns=[target_column[0]])
y_df = df[target_column[0]]

X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_df, test_size=0.2, random_state=42)

original_labels = None
if y_df.dtype == 'object' or y_df.dtype.name == 'category':
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)
    original_labels = encoder.classes_

max_k = min(max_k, X_train.shape[1] - len(deleted_columns))

# Suprime os avisos específicos do Optuna
warnings.filterwarnings(
    "ignore", message="Choices for a categorical distribution should be a tuple of None, bool, int, float and str")


def objective(trial):
    # Hiperparâmetro para o escalonador
    scaler_option = trial.suggest_categorical('scaler', scalers)

    # Escolha do escalonador baseado no hiperparâmetro
    if scaler_option == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaler_option == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_option == 'RobustScaler':
        scaler = RobustScaler()

    # Hiperparâmetros do SelectKBest
    k = trial.suggest_int('kbest__k', min_k, max_k)
    score_func_option = trial.suggest_categorical(
        'kbest__score_func', score_functions)

    if score_func_option == 'f_classif':
        score_func = f_classif
    elif score_func_option == 'chi2':
        score_func = chi2
    elif score_func_option == 'mutual_info_classif':
        score_func = mutual_info_classif

    # Configuração do preprocessor
    preprocessor = ColumnTransformer([
        ('deleter', 'drop', deleted_columns),
        ('scaler', scaler, [
         col for col in X_train.columns if col not in deleted_columns])
    ])

    # Criação do modelo baseado na seleção
    if model_selection == "MLP Classifier":
        # Hiperparâmetros do MLP
        hidden_layer_sizes = trial.suggest_categorical(
            'mlp__hidden_layer_sizes', hidden_layer_size)
        alpha = trial.suggest_float('mlp__alpha', min_alfa, max_alfa, log=True)
        learning_rate_init = trial.suggest_float(
            'mlp__learning_rate_init', min_learning_rate_init, max_learning_rate_init, log=True)
        activation = trial.suggest_categorical(
            'mlp__activation', activation_functions)
        solver = trial.suggest_categorical('mlp__solver', solvers)
        learning_rate = trial.suggest_categorical(
            'mlp__learning_rate', learning_rates)
        max_iter = trial.suggest_int(
            'mlp__max_iter', min_epochs, max_epochs)

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            activation=activation,
            solver=solver,
            max_iter=max_iter,
            learning_rate=learning_rate,
            random_state=42,
        )

        pipeline = SKPipeline([
            ('preprocessor', preprocessor),
            ('imputer', SimpleImputer(strategy='mean')),
            ('kbest', SelectKBest(score_func=score_func, k=k)),
            ('mlp', model)
        ])

    elif model_selection == "Random Forest":
        # Hiperparâmetros do Random Forest
        n_estimators = trial.suggest_int(
            'rf__n_estimators', min_n_estimators, max_n_estimators)
        max_depth = trial.suggest_int(
            'rf__max_depth', min_max_depth, max_max_depth)
        min_samples_leaf = trial.suggest_int(
            'rf__min_samples_leaf', min_min_samples_leaf, max_min_samples_leaf)
        min_samples_split = trial.suggest_int(
            'rf__min_samples_split', min_min_samples_split, max_min_samples_split)
        max_features = trial.suggest_categorical(
            'rf__max_features', max_features_options)
        criterion = trial.suggest_categorical(
            'rf__criterion', criterion_options)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            criterion=criterion,
            random_state=42,
        )

        pipeline = SKPipeline([
            ('preprocessor', preprocessor),
            ('imputer', SimpleImputer(strategy='mean')),
            ('kbest', SelectKBest(score_func=score_func, k=k)),
            ('rf', model)
        ])

    # Validação cruzada com tratamento de erro para chi2
    try:
        cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv,
                                 scoring=score_metric, n_jobs=-1)
        return scores.mean()
    except ValueError as e:
        # Ignora erros do chi2 relacionados a valores negativos
        if 'Input X must be non-negative' in str(e):
            return float('-inf')  # Indica falha no trial
        else:
            raise e


# Estudo com Optuna
study = optuna.create_study(direction='maximize')


def logging_callback(study, trial):
    progress_msg = f"[PROGRESS] Trial {trial.number} finished with value: {trial.value} and parameters: {trial.params}. Best is trial {study.best_trial.number} with value: {study.best_value}.\n"
    print(progress_msg, file=sys.stderr, flush=True)


# Número de combinações a testar
study.optimize(objective, n_trials=num_trials, callbacks=[logging_callback])

# Configurando o pipeline com os melhores hiperparâmetros encontrados
best_params = study.best_params

# Inicializando o pipeline com os melhores parâmetros
scaler_option = best_params['scaler']
if scaler_option == 'MinMaxScaler':
    scaler = MinMaxScaler()
elif scaler_option == 'StandardScaler':
    scaler = StandardScaler()
elif scaler_option == 'RobustScaler':
    scaler = RobustScaler()

score_func_option = best_params["kbest__score_func"]
if score_func_option == 'f_classif':
    score_func = f_classif
elif score_func_option == 'chi2':
    score_func = chi2
elif score_func_option == 'mutual_info_classif':
    score_func = mutual_info_classif

preprocessor = ColumnTransformer([
    ('deleter', 'drop', deleted_columns),
    ('scaler', scaler, [
     col for col in X_train.columns if col not in deleted_columns])
])

# Criação do pipeline final baseado na seleção do modelo
if model_selection == "MLP Classifier":
    model_final = MLPClassifier(
        hidden_layer_sizes=best_params['mlp__hidden_layer_sizes'],
        alpha=best_params['mlp__alpha'],
        learning_rate_init=best_params['mlp__learning_rate_init'],
        activation=best_params['mlp__activation'],
        solver=best_params['mlp__solver'],
        max_iter=best_params['mlp__max_iter'],
        learning_rate=best_params['mlp__learning_rate'],
        random_state=42,
        verbose=True  # Para capturar o progresso do treinamento
    )

    pipeline_final = SKPipeline([
        ('preprocessor', preprocessor),
        ('imputer', SimpleImputer(strategy='mean')),
        ('kbest', SelectKBest(
            score_func=score_func, k=best_params['kbest__k'])),
        ('mlp', model_final)
    ])

elif model_selection == "Random Forest":
    model_final = RandomForestClassifier(
        n_estimators=best_params['rf__n_estimators'],
        max_depth=best_params['rf__max_depth'],
        min_samples_leaf=best_params['rf__min_samples_leaf'],
        min_samples_split=best_params['rf__min_samples_split'],
        max_features=best_params['rf__max_features'],
        criterion=best_params['rf__criterion'],
        random_state=42,
    )

    pipeline_final = SKPipeline([
        ('preprocessor', preprocessor),
        ('imputer', SimpleImputer(strategy='mean')),
        ('kbest', SelectKBest(
            score_func=score_func, k=best_params['kbest__k'])),
        ('rf', model_final)
    ])


# Treinando o modelo
class TrainingVerboseCapture:
    def __init__(self, model_type):
        self.loss_values = []
        self.iteration = 0
        self.original_stdout = None
        self.captured_output = StringIO()
        self.model_type = model_type

    def start_capture(self):
        """Inicia a captura do stdout"""
        self.original_stdout = sys.stdout
        sys.stdout = self.captured_output

    def stop_capture(self):
        """Para a captura e processa a saída"""
        if self.original_stdout:
            sys.stdout = self.original_stdout
            self.process_output()

    def process_output(self):
        """Processa a saída capturada para extrair valores de loss"""
        output_text = self.captured_output.getvalue()
        lines = output_text.split('\n')

        for line in lines:
            if self.model_type == "MLP Classifier" and 'Iteration' in line and 'loss' in line:
                # Extrai o valor da loss da saída verbose do MLP
                loss_match = re.search(r'loss = ([\d.]+)', line)
                if loss_match:
                    loss_value = float(loss_match.group(1))
                    self.loss_values.append(loss_value)
                    self.iteration += 1

                    # Envia progresso via stderr para o WebSocket
                    verbose_msg = f"[TRAINING] Iteration {self.iteration}, Loss: {loss_value:.6f}\n"
                    print(verbose_msg, file=sys.stderr, flush=True)
                    time.sleep(0.2)
            elif self.model_type == "Random Forest" and ('building tree' in line.lower() or 'tree' in line.lower()):
                # Para Random Forest, captura informações sobre construção de árvores
                verbose_msg = f"[TRAINING] {line.strip()}\n"
                print(verbose_msg, file=sys.stderr, flush=True)


# Captura o verbose do modelo
verbose_capture = TrainingVerboseCapture(model_selection)

print(f"[TRAINING] Starting model training...\n",
      file=sys.stderr, flush=True)

# Inicia captura apenas para MLP (Random Forest não precisa de captura de stdout)
if model_selection == "MLP Classifier":
    verbose_capture.start_capture()

try:
    # Treina o modelo
    pipeline_final.fit(X_train, y_train)
finally:
    # Para a captura mesmo se houver erro (apenas para MLP)
    if model_selection == "MLP Classifier":
        verbose_capture.stop_capture()

print(f"[TRAINING] Training completed!\n",
      file=sys.stderr, flush=True)

# Gerar predições no conjunto de teste
print("[EVALUATION] Generating predicts on the test set...\n",
      file=sys.stderr, flush=True)
y_pred = pipeline_final.predict(X_test)

# Gerar o classification_report
if original_labels is not None:
    # Se temos labels codificados, usar os nomes originais
    class_report = classification_report(
        y_test, y_pred, target_names=original_labels, output_dict=True)
else:
    # Se não temos encoder, usar os valores numéricos diretamente
    class_report = classification_report(y_test, y_pred, output_dict=True)

print("[EVALUATION] Classification report generated!\n",
      file=sys.stderr, flush=True)

model_with_metadata = ModelMetadata(
    pipeline=pipeline_final,
    target_column=target_column,
    classification_report=class_report
)


# Resultados
def serialize_best_params(best_params):
    serializable_params = {}
    for key, value in best_params.items():
        if callable(value):
            serializable_params[key] = value.__name__
        else:
            serializable_params[key] = value
    return serializable_params


result = {
    "model_type": model_selection,
    "best_params": serialize_best_params(study.best_params),
    "best_value": study.best_value,
    "classification_report": class_report
}

joblib.dump(model_with_metadata, f"public/model/model.pkl")
print("[DONE] Study Completed.\n", file=sys.stderr, flush=True)

# Converte para JSON
print(json.dumps(result, indent=4))
