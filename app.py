import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score

from utils.data_processing import load_and_preprocess_data

# --- Configuração Inicial ---
st.set_page_config(page_title="Painel de Vacinação", layout="wide")
st.title("Painel de Vacinação Interativo")
st.markdown("Explore, analise e projete tendências futuras de vacinação.")

# --- Carregamento de dados ---
@st.cache_data
def load_data(file_path):
    df = load_and_preprocess_data(file_path)
    df['Data da Vacinação'] = pd.to_datetime(df['Data da Vacinação'], errors='coerce')
    df = df.dropna(subset=['Data da Vacinação'])
    df = df[df['Sexo'].isin(['FEMININO', 'MASCULINO'])]
    df = df[df['Raça'].isin(['BRANCA', 'PRETA', 'PARDA', 'AMARELA', 'INDÍGENA'])]
    df = df[df['Idade'] > 0]
    return df

FILE_PATH = "vacinados.csv"
df = load_data(FILE_PATH)

# --- Seleção de módulo ---
module = st.sidebar.radio(
    "Escolha um módulo:",
    ("Exploração de Dados", "Análise Histórica", "Machine Learning", "Predição Futuras Vacinações")
)

# --- 1. Exploração de Dados ---
if module == "Exploração de Dados":
    st.header("Exploração de Dados")
    
    # Filtros principais
    col1, col2 = st.columns(2)
    filtro_raca = col1.selectbox("Raça", ["Todas"] + sorted(df['Raça'].unique()))
    filtro_fab = col2.selectbox("Fabricante da Vacina", ["Todos"] + sorted(df['Fabricante da Vacina'].unique()))
    
    # Filtros adicionais
    col3, col4, col5 = st.columns(3)
    filtro_sexo = col3.radio("Sexo", ["Todos"] + sorted(df['Sexo'].unique()))
    filtro_idade = col4.slider("Faixa Etária", int(df['Idade'].min()), int(df['Idade'].max()), (0, 100))
    filtro_categoria = col5.multiselect("Categoria", sorted(df['Categoria'].unique()))
    
    # Aplicando filtros
    df_filtered = df.copy()
    if filtro_raca != "Todas":
        df_filtered = df_filtered[df_filtered['Raça'] == filtro_raca]
    if filtro_fab != "Todos":
        df_filtered = df_filtered[df_filtered['Fabricante da Vacina'] == filtro_fab]
    if filtro_sexo != "Todos":
        df_filtered = df_filtered[df_filtered['Sexo'] == filtro_sexo]
    df_filtered = df_filtered[(df_filtered['Idade'] >= filtro_idade[0]) & (df_filtered['Idade'] <= filtro_idade[1])]
    if filtro_categoria:
        df_filtered = df_filtered[df_filtered['Categoria'].isin(filtro_categoria)]
    
    st.subheader("Tabela filtrada (top 20)")
    st.dataframe(df_filtered.head(20))

# --- 2. Análise Histórica ---
elif module == "Análise Histórica":
    st.header("Análise Histórica")
    
    # Filtros minimalistas
    col1, col2, col3 = st.columns(3)
    sexo_sel = col1.radio("Sexo", ["Todos"] + sorted(df['Sexo'].unique()))
    idade_sel = col2.slider("Faixa Etária", int(df['Idade'].min()), int(df['Idade'].max()), (0, 100))
    raca_sel = col3.multiselect("Raça", sorted(df['Raça'].unique()), default=sorted(df['Raça'].unique()))
    
    df_hist = df.copy()
    if sexo_sel != "Todos":
        df_hist = df_hist[df_hist['Sexo'] == sexo_sel]
    df_hist = df_hist[(df_hist['Idade'] >= idade_sel[0]) & (df_hist['Idade'] <= idade_sel[1])]
    df_hist = df_hist[df_hist['Raça'].isin(raca_sel)]
    
    # Gráficos principais
    col1, col2 = st.columns(2)
    sexo_counts = df_hist['Sexo'].value_counts().reset_index()
    sexo_counts.columns = ['Sexo', 'Contagem']
    fig_sexo = px.bar(sexo_counts, x='Sexo', y='Contagem', color='Sexo', title="Distribuição por Sexo")
    if sexo_sel == "Todos":
        col1.plotly_chart(fig_sexo, use_container_width=True)
    
    raca_counts = df_hist['Raça'].value_counts().reset_index()
    raca_counts.columns = ['Raça', 'Contagem']
    fig_raca = px.pie(raca_counts, names='Raça', values='Contagem', hole=0.3, title="Distribuição por Raça")
    col2.plotly_chart(fig_raca, use_container_width=True)
    
    fab_counts = df_hist['Fabricante da Vacina'].value_counts().reset_index()
    fab_counts.columns = ['Fabricante', 'Contagem']
    fig_fab = px.pie(fab_counts, names='Fabricante', values='Contagem', hole=0.3, title="Distribuição por Fabricante")
    st.plotly_chart(fig_fab, use_container_width=True)
    
    idade_hist = df_hist.groupby('Idade').size().reset_index(name='Contagem')
    fig_idade = px.line(idade_hist, x='Idade', y='Contagem', markers=True, title="Vacinações por Faixa Etária")
    st.plotly_chart(fig_idade, use_container_width=True)

# --- 3. Machine Learning (Clusterização) ---
elif module == "Machine Learning":
    st.header("Clusterização de Casos (K-Means)")
    
    n_clusters = st.slider("Número de Clusters (K)", min_value=2, max_value=8, value=3)
    
    df_ml = df.copy()
    # Converter colunas categóricas em dummies
    df_ml = pd.get_dummies(df_ml, columns=['Sexo', 'Raça', 'Fabricante da Vacina'], drop_first=True)
    
    # Selecionar apenas colunas numéricas
    X_kmeans = df_ml.select_dtypes(include=np.number)
    
    if st.button(f"Aplicar K-Means com {n_clusters} Clusters"):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_kmeans)
        
        df_clustered = df.copy()
        df_clustered['Cluster'] = clusters
        
        st.success("Clusterização realizada!")
        
        # Perfil dos clusters
        cluster_profile = df_clustered.groupby('Cluster').agg(
            idade_media=('Idade', 'mean'),
            contagem=('Idade', 'count')
        ).round(2)
        st.subheader("Perfil dos Clusters")
        st.dataframe(cluster_profile)
        
        # Visualização cluster idade média
        fig_cluster = px.bar(
            cluster_profile.reset_index(),
            x='Cluster',
            y='idade_media',
            title="Idade Média por Cluster",
            labels={'idade_media':'Idade Média'}
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

# --- 4. Predição Futuras Vacinações ---
elif module == "Predição Futuras Vacinações":
    st.header("Projeção Futuras Vacinações")
    
    col1, col2 = st.columns(2)
    horizon = col1.slider("Meses à frente", 1, 36, 12)
    sexo_proj = col2.multiselect("Sexo", ["Todos"] + sorted(df['Sexo'].unique()), default=["Todos"])
    
    df_pred = df.copy()
    if "Todos" not in sexo_proj:
        df_pred = df_pred[df_pred['Sexo'].isin(sexo_proj)]
    
    df_pred['Mes'] = df_pred['Data da Vacinação'].dt.to_period('M')
    df_month = df_pred.groupby('Mes').size().reset_index(name='Vacinações')
    
    df_month['Mes_Num'] = np.arange(len(df_month))
    X = df_month[['Mes_Num']]
    y = df_month['Vacinações']
    
    model = LinearRegression()
    model.fit(X, y)
    
    X_future = np.arange(len(df_month), len(df_month) + horizon).reshape(-1,1)
    y_future = model.predict(X_future)
    
    future_dates = pd.period_range(df_month['Mes'].iloc[-1] + 1, periods=horizon, freq='M')
    df_proj = pd.DataFrame({'Mes': future_dates.astype(str), 'Vacinações Previstas': y_future})
    
    fig_proj = px.line(df_proj, x='Mes', y='Vacinações Previstas', markers=True,
                       title="Projeção de Vacinações Futuras")
    st.plotly_chart(fig_proj, use_container_width=True)
