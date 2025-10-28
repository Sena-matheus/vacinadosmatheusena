import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import warnings
import re
import streamlit as st 

warnings.filterwarnings('ignore')

def clean_column_names(df):
    new_cols = []
    for col in df.columns:
        clean_col = str(col).lower()
        clean_col = re.sub(r'[^a-z0-9_]', '', clean_col)
        new_cols.append(clean_col)
    df.columns = new_cols
    return df

def load_and_preprocess_data(file_path):
    df = pd.DataFrame()
    
    AMOSTRA_LIMITE = 50000 
    
    tentativas = [(';', 'latin1'), (',', 'utf-8'), (',', 'latin1')]

    for sep, encoding in tentativas:
        try:
            df = pd.read_csv(file_path, sep=sep, encoding=encoding, 
                             low_memory=False, nrows=AMOSTRA_LIMITE) 
            if not df.empty and len(df.columns) > 1:
                st.info(f"Carregamento bem-sucedido. Usando apenas {len(df)} linhas para performance.")
                break
            df = pd.DataFrame()
        except Exception:
            continue
    
    if df.empty or len(df.columns) <= 1:
        raise ValueError(f"Não foi possível carregar o dataset '{file_path}'. Verifique o formato e a localização.")
        
    df = clean_column_names(df)
    
    cols_selecionadas = [
        'idade',
        'sexo',
        'raca_cor',
        'vacina_fabricante',
        'descricao_dose'
    ]

    missing_cols = [col for col in cols_selecionadas if col not in df.columns]
    if missing_cols:
        raise Exception(f"As seguintes colunas estão faltando/erradas: {missing_cols}. Colunas disponíveis: {df.columns.tolist()}")

    df = df[cols_selecionadas].copy()
    
    df['descricao_dose'] = df['descricao_dose'].astype(str).str.strip().str.upper()
    df.dropna(subset=['descricao_dose'], inplace=True)
    
    top_doses = df['descricao_dose'].value_counts().nlargest(4).index
    df = df[df['descricao_dose'].isin(top_doses)].copy()
    
    le = LabelEncoder()
    df['alvo_classificacao'] = le.fit_transform(df['descricao_dose'])
    df['idade'] = pd.to_numeric(df['idade'], errors='coerce')
    df['idade'].fillna(df['idade'].median(), inplace=True)
    
    for col in ['sexo', 'raca_cor', 'vacina_fabricante']:
        df[col] = df[col].astype(str).str.strip()
        df[col].fillna(df[col].mode()[0], inplace=True)
        
    df.drop(columns=['descricao_dose'], inplace=True)
    df.rename(columns={'alvo_classificacao': 'tipo_de_dose'}, inplace=True)

    return df

def create_preprocessor(df, target_column):
    """
    Cria o ColumnTransformer (Pipeline de Pré-processamento).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    numerical_features = ['idade'] 
    categorical_features = ['sexo', 'raca_cor', 'vacina_fabricante'] # 'municipio' removido

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    return X, y, preprocessor