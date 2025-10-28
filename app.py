import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.express as px
import numpy as np

from utils.data_processing import load_and_preprocess_data, create_preprocessor
from utils.models import train_supervised_model, evaluate_supervised_model, apply_unsupervised_model


# --- Configuração Inicial e Carregamento (Cache para Performance) ---
st.set_page_config(
    page_title="ML na Saúde - Dados de Vacinação",
    layout="wide"
)

st.title("💉 Machine Learning: Classificação de Doses de Vacina")
st.markdown("Aplicação interativa em Streamlit para prever o tipo de dose (1ª, 2ª, Reforço, etc.) com base nas features do paciente e da vacina.")

FILE_PATH = 'vacinados.csv' # NOVO NOME DO ARQUIVO
TARGET_COLUMN = 'tipo_de_dose' # Alvo renomeado em data_processing.py (Label Encoding)

@st.cache_data
def load_and_prepare_data(file_path, target_column):
    """Função única para carregar, limpar, e aplicar o pré-processamento inicial."""
    df = load_and_preprocess_data(file_path)
    X, y, preprocessor = create_preprocessor(df, target_column)

    # 80/20 Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Aplicar o pré-processamento
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    X_processed = preprocessor.transform(X)

    return df, X_train_processed, X_test_processed, y_train, y_test, X_processed, preprocessor

try:
    df, X_train_p, X_test_p, y_train, y_test, X_processed, preprocessor = load_and_prepare_data(FILE_PATH, TARGET_COLUMN)
    feature_names = preprocessor.get_feature_names_out() 
except Exception as e:
    st.error(f"❌ Erro Crítico: Falha no carregamento ou pré-processamento. Detalhe: {e}")
    st.stop()


# --- Sidebar para Navegação e Controles ---
st.sidebar.title("🛠️ Configurações e Análises")
analysis_type = st.sidebar.radio(
    "Selecione o Módulo de Análise:",
    ("1. Análise Exploratória", "2. Aprendizagem Supervisionada", "3. Aprendizagem Não Supervisionada")
)

# --- 1. Análise Exploratória de Dados (EDA) ---
if analysis_type == "1. Análise Exploratória":
    st.header("📊 Módulo 1: Análise Exploratória de Dados (EDA)")
    
    st.subheader("Amostra dos Dados")
    st.dataframe(df.head())
    st.write(f"**Shape do Dataset (Amostras, Features):** {df.shape}")
    
    # Gráfico da Distribuição da Variável Alvo
    st.subheader(f"Distribuição da Variável Alvo ({TARGET_COLUMN})")
    st.info("A variável alvo foi codificada para 0, 1, 2, 3... (Label Encoding) para as doses mais comuns.")
    
    target_counts = df[TARGET_COLUMN].value_counts().reset_index()
    target_counts.columns = ['Dose_Codificada', 'Contagem']
    fig = px.bar(target_counts, x='Dose_Codificada', y='Contagem', color='Dose_Codificada', 
                 title="Equilíbrio da Variável Alvo (Tipos de Dose Codificados)")
    st.plotly_chart(fig, use_container_width=True)


# --- 2. Aprendizagem Supervisionada (Classificação) ---
elif analysis_type == "2. Aprendizagem Supervisionada":
    st.header("⚙️ Módulo 2: Classificação Multiclasse (Regressão Logística)")

    if st.button("🚀 Treinar e Avaliar Modelo de Classificação"):
        with st.spinner('Treinando Modelo e avaliando...'):
            model = train_supervised_model(X_train_p, y_train)
            metrics = evaluate_supervised_model(model, X_test_p, y_test)
            
            st.success("Modelo Treinado e Avaliado com Sucesso!")
            
            st.subheader("Resultados de Avaliação (Conjunto de Teste)")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Acurácia", f"{metrics['Acurácia']:.4f}")
                
            with col2:
                report_df = pd.DataFrame(metrics['Relatório de Classificação']).transpose()
                st.markdown("**Principais Métricas (Suporte Multiclasse)**")
                st.dataframe(report_df.round(4))
            
# --- 3. Aprendizagem Não Supervisionada (Clusterização) ---
elif analysis_type == "3. Aprendizagem Não Supervisionada":
    st.header("💡 Módulo 3: Segmentação de Casos (K-Means)")
    
    n_clusters = st.sidebar.slider("Selecione o número de Clusters (K):", min_value=2, max_value=8, value=3)
    
    if st.button(f"✨ Aplicar K-Means com K={n_clusters}"):
        with st.spinner(f'Aplicando K-Means com {n_clusters} clusters...'):
            clusters, centers, score = apply_unsupervised_model(X_processed, n_clusters)
            
            df_clustered = df.copy()
            df_clustered['Cluster'] = clusters
            
            st.success("Clusterização Concluída!")
            st.write(f"**Silhouette Score:** `{score:.4f}`")
            
            st.subheader("Análise do Perfil dos Clusters")
            
            cluster_profile = df_clustered.groupby('Cluster').agg(
                idade_media=('idade', 'mean'),
                media_tipo_dose=(TARGET_COLUMN, 'mean'),
                contagem=('idade', 'count')
            ).round(3)
            
            st.dataframe(cluster_profile.style.format({
                'media_tipo_dose': '{:.2f}',
                'idade_media': '{:.1f}',
                'contagem': '{:}'
            }))