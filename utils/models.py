from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score, classification_report
import pandas as pd

# --- Modelo Supervisionado (Regressão Logística para Classificação) ---
def train_supervised_model(X_train_processed, y_train):
    """Treina um modelo de Regressão Logística (suporta multiclasse)."""
    # 'lbfgs' é o solver padrão e funciona bem para multiclasse ('multinomial')
    model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs', multi_class='auto') 
    model.fit(X_train_processed, y_train)
    return model

def evaluate_supervised_model(model, X_test_processed, y_test):
    """Avalia o modelo e retorna métricas chave."""
    y_pred = model.predict(X_test_processed)
    
    metrics = {
        'Acurácia': accuracy_score(y_test, y_pred),
        # Usamos average='weighted' para classification_report em multiclasse
        'Relatório de Classificação': classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    }
    return metrics

# --- Modelo Não Supervisionado (K-Means para Clusterização) ---
# (Permanece inalterado, é genérico)
def apply_unsupervised_model(X_processed, n_clusters):
    """Aplica o K-Means e retorna os clusters e o score."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, init='k-means++')
    clusters = kmeans.fit_predict(X_processed)
    
    try:
        score = silhouette_score(X_processed, clusters)
    except ValueError:
        score = -1 
        
    return clusters, kmeans.cluster_centers_, score