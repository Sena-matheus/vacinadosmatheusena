from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score, classification_report, mean_squared_error, r2_score
import pandas as pd
import numpy as np


# --- Modelo Supervisionado (Classificação de Doses) ---
def train_supervised_model(X_train_processed, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs', multi_class='auto')
    model.fit(X_train_processed, y_train)
    return model


def evaluate_supervised_model(model, X_test_processed, y_test):
    y_pred = model.predict(X_test_processed)
    metrics = {
        'Acurácia': accuracy_score(y_test, y_pred),
        'Relatório de Classificação': classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    }
    return metrics


# --- Modelo Não Supervisionado (Clusterização com K-Means) ---
def apply_unsupervised_model(df, n_clusters=3, categorical_cols=None, feature_cols=None):
    """
    Aplica K-Means sobre df, convertendo colunas categóricas em dummies
    e selecionando apenas colunas numéricas para o modelo.
    
    Args:
        df: pd.DataFrame original
        n_clusters: número de clusters
        categorical_cols: lista de colunas categóricas para transformar em dummies
        feature_cols: lista de colunas numéricas adicionais
    
    Returns:
        df_clustered: DataFrame com coluna 'Cluster'
        cluster_centers: array dos centros do K-Means
        silhouette_score: pontuação do Silhouette
    """
    df_ml = df.copy()
    
    # Criar dummies das colunas categóricas
    if categorical_cols:
        df_ml = pd.get_dummies(df_ml, columns=categorical_cols, drop_first=True)
    
    # Selecionar colunas numéricas
    if feature_cols:
        X = df_ml[feature_cols + [c for c in df_ml.columns if c not in feature_cols and df_ml[c].dtype in [np.int64, np.float64]]]
    else:
        X = df_ml.select_dtypes(include=np.number)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    try:
        score = silhouette_score(X, clusters)
    except ValueError:
        score = -1  # caso haja apenas um cluster ou dados homogêneos
    
    df_clustered = df.copy()
    df_clustered['Cluster'] = clusters
    
    return df_clustered, kmeans.cluster_centers_, score


# --- Modelo Preditivo (Regressão Linear Temporal) ---
def train_predictive_model(df, feature_col='Idade', target_col='Tipo de Dose'):
    """
    Regressão Linear simples para prever tendências de vacinação futuras.
    """
    X = df[[feature_col]].copy()
    y = df[target_col].copy()
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    metrics = {'RMSE': rmse, 'R2': r2}
    
    return model, metrics, y_pred