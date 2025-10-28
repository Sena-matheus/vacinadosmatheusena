import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import warnings
import re
import streamlit as st
from utils.config import AMOSTRA_LIMITE

warnings.filterwarnings('ignore')


# --- Limpeza de nomes das colunas ---
def clean_column_names(df):
    """Padroniza nomes das colunas para minúsculas e remove caracteres inválidos."""
    new_cols = []
    for col in df.columns:
        clean_col = str(col).lower().strip()
        clean_col = re.sub(r'[^a-z0-9_]', '_', clean_col)
        new_cols.append(clean_col)
    df.columns = new_cols
    return df


# --- Carregamento e pré-processamento principal ---
def load_and_preprocess_data(file_path):
    """Carrega, limpa e pré-processa os dados de vacinação."""
    df = pd.DataFrame()
    tentativas = [(';', 'latin1'), (',', 'utf-8'), (',', 'latin1')]

    for sep, encoding in tentativas:
        try:
            df = pd.read_csv(file_path, sep=sep, encoding=encoding,
                             low_memory=False, nrows=AMOSTRA_LIMITE)
            if not df.empty and len(df.columns) > 1:
                st.info(f"✅ Dados carregados com sucesso ({len(df)} linhas).")
                break
            df = pd.DataFrame()
        except Exception:
            continue

    if df.empty or len(df.columns) <= 1:
        raise ValueError(f"❌ Não foi possível carregar o dataset '{file_path}'.")

    # --- Limpeza e padronização ---
    df = clean_column_names(df)

    colunas_necessarias = [
        'faixa_etaria', 'idade', 'sexo', 'raca_cor', 'municipio',
        'grupo', 'categoria', 'lote', 'vacina_fabricante',
        'descricao_dose', 'data_vacinacao'
    ]

    faltando = [c for c in colunas_necessarias if c not in df.columns]
    if faltando:
        raise Exception(f"As colunas a seguir estão ausentes: {faltando}")

    df = df[colunas_necessarias].copy()

    # --- Renomear para nomes mais formais ---
    df.rename(columns={
        'faixa_etaria': 'Faixa Etária',
        'idade': 'Idade',
        'sexo': 'Sexo',
        'raca_cor': 'Raça',
        'municipio': 'Município',
        'grupo': 'Grupo',
        'categoria': 'Categoria',
        'lote': 'Lote',
        'vacina_fabricante': 'Fabricante da Vacina',
        'descricao_dose': 'Descrição da Dose',
        'data_vacinacao': 'Data da Vacinação'
    }, inplace=True)

    # --- Limpeza de valores ---
    df['Descrição da Dose'] = df['Descrição da Dose'].astype(str).str.upper().str.strip()
    df['Idade'] = pd.to_numeric(df['Idade'], errors='coerce')
    df['Idade'].fillna(df['Idade'].median(), inplace=True)

    for col in ['Sexo', 'Raça', 'Fabricante da Vacina', 'Município', 'Categoria']:
        df[col] = df[col].astype(str).str.strip().str.upper()
        df[col].replace(['', 'NAN', 'NULL'], np.nan, inplace=True)
        df[col].fillna(df[col].mode()[0], inplace=True)

    # --- Criar variável alvo para ML (descrição da dose) ---
    le = LabelEncoder()
    df['Tipo de Dose'] = le.fit_transform(df['Descrição da Dose'])

    return df


# --- Pré-processador (para ML) ---
def create_preprocessor(df, target_column):
    """Cria o transformador de colunas para treino de modelo."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    numeric_features = ['Idade']
    categorical_features = ['Sexo', 'Raça', 'Fabricante da Vacina', 'Município', 'Categoria']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    return X, y, preprocessor