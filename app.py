import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import unicodedata

from utils.data_processing import load_and_preprocess_data

def strip_accents(text: str) -> str:
    if not isinstance(text, str):
        return text
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")

def normalize_col_name(name: str) -> str:
    return strip_accents(name).strip().lower().replace(" ", "_")

def find_col(df: pd.DataFrame, candidates):
    """
    Encontra o primeiro nome de coluna presente em df entre os candidatos.
    Aceita candidatos com ou sem acentuação. Retorna None se não encontrar.
    """
    cols_norm = {normalize_col_name(c): c for c in df.columns}
    for cand in candidates:
        cand_norm = normalize_col_name(cand)
        if cand_norm in cols_norm:
            return cols_norm[cand_norm]  # retorna nome exato do df
    return None

def safe_series_str_normalize(s: pd.Series):
    # garante que strings fiquem sem acento para comparações internas, sem perder conteúdo original
    return s.astype(str).map(lambda x: strip_accents(x).upper().strip())

st.set_page_config(page_title="Painel de Vacinação", layout="wide")
st.title("Painel Interativo de Vacinação")
st.markdown("Visualize dados, tendências e previsões de vacinação de forma interativa e profissional.")

# CSS para blocos de notícias
st.markdown("""
    <style>
    h1, h2, h3 { color: #004d40; }
    .news-block {
        background-color: #f8f9fa;
        border-left: 4px solid #004d40;
        padding: 10px 15px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file_path):
    """
    Tenta carregar via helper load_and_preprocess_data; caso falhe na decodificação,
    faz fallback para diversas codificações e normaliza datas.
    """
    try:
        df = load_and_preprocess_data(file_path)
    except Exception:
        # fallback genérico
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except Exception:
            df = pd.read_csv(file_path, encoding="latin-1", sep=None, engine="python")

    # strip de nomes de colunas (mantemos os nomes originais mas removemos espaços extremos)
    df.columns = [c.strip() for c in df.columns]

    # normalizar strings nas colunas object para evitar caracteres bizarros
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).map(lambda x: x.strip())

    # identificar colunas chave (aceita diversas grafias)
    col_data = find_col(df, ["Data da Vacinação", "Data da Vacinacao", "data_da_vacinacao", "data_vacinacao", "data"])
    col_sexo = find_col(df, ["Sexo", "sexo"])
    col_raca = find_col(df, ["Raça", "Raca", "raca"])
    col_idade = find_col(df, ["Idade", "idade", "idade_anos"])
    col_fab = find_col(df, ["Fabricante da Vacina", "Fabricante", "fabricante_da_vacina", "fabricante"])

    # converter data se existir
    if col_data:
        df[col_data] = pd.to_datetime(df[col_data], errors="coerce")

    # filtrar linhas inválidas
    if col_data:
        df = df.dropna(subset=[col_data])

    # segurança: manter colunas esperadas como strings/consistentes
    if col_sexo:
        # normalizar valores de sexo (sem acento, uppercase)
        df[col_sexo] = df[col_sexo].astype(str).map(lambda x: strip_accents(x).upper().strip())
    if col_raca:
        df[col_raca] = df[col_raca].astype(str).map(lambda x: strip_accents(x).upper().strip())
    if col_fab:
        df[col_fab] = df[col_fab].astype(str).map(lambda x: x.strip())
    if col_idade:
        # garantir numérico
        df[col_idade] = pd.to_numeric(df[col_idade], errors="coerce").fillna(0).astype(int)

    # Retornar DataFrame e nomes das colunas detectadas para uso posterior
    return df, {
        "data": col_data,
        "sexo": col_sexo,
        "raca": col_raca,
        "idade": col_idade,
        "fabricante": col_fab
    }

FILE_PATH = "vacinados.csv"
df, COLS = load_data(FILE_PATH)

# validação rápida: se colunas essenciais não existirem, avisar o usuário
essential = ["data", "sexo", "raca", "idade", "fabricante"]
missing = [k for k in essential if COLS.get(k) is None]
if missing:
    st.error(f"Colunas essenciais não encontradas no arquivo: {missing}. Verifique nomes do CSV.")
    st.stop()

# atalhos para nomes reais
COL_DATA = COLS["data"]
COL_SEXO = COLS["sexo"]
COL_RACA = COLS["raca"]
COL_IDADE = COLS["idade"]
COL_FAB = COLS["fabricante"]

# garantir filtros preditivos/plotly: criar col com periodo mensal
df["_mes_periodo"] = df[COL_DATA].dt.to_period("M")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de registros", f"{len(df):,}".replace(",", "."))
col2.metric("Idade média", round(df[COL_IDADE].mean(), 1))
# fabricante e sexo podem ter valores NaN se coluna vazia; segurança:
fab_mode = df[COL_FAB].mode().iloc[0] if not df[COL_FAB].mode().empty else "N/A"
sexo_mode = df[COL_SEXO].mode().iloc[0] if not df[COL_SEXO].mode().empty else "N/A"
col3.metric("Fabricante mais aplicado", fab_mode)
col4.metric("Sexo predominante", sexo_mode)

st.markdown("---")

module = st.sidebar.radio(
    "Escolha uma seção:",
    (
        "Visão Geral",
        "Tendências e Histórico",
        "Agrupamentos Inteligentes",
        "Projeções Futuras"
    )
)

if module == "Visão Geral":
    st.header("Visão Geral dos Dados")
    st.caption("Use os filtros para refinar a visualização e gerar insights em formato de notícias.")

    # filtros principais (usando nomes detectados)
    col1, col2 = st.columns(2)
    filtro_raca = col1.selectbox("Raça", ["Todas"] + sorted(df[COL_RACA].unique()))
    filtro_fab = col2.selectbox("Fabricante", ["Todos"] + sorted(df[COL_FAB].unique()))

    col3, col4, col5 = st.columns(3)
    filtro_sexo = col3.radio("Sexo", ["Todos"] + sorted(df[COL_SEXO].unique()))
    filtro_idade = col4.slider("Faixa etária", int(df[COL_IDADE].min()), int(df[COL_IDADE].max()), (0, 100))
    # categoria é opcional — só usar se existir na base
    categoria_col = find_col(df, ["Categoria", "categoria"])
    categorias = sorted(df[categoria_col].unique()) if categoria_col else []
    filtro_categoria = col5.multiselect("Categoria", categorias) if categorias else []

    # aplicar filtros ao df_filtered
    df_filtered = df.copy()
    if filtro_raca and filtro_raca != "Todas":
        df_filtered = df_filtered[df_filtered[COL_RACA] == filtro_raca]
    if filtro_fab and filtro_fab != "Todos":
        df_filtered = df_filtered[df_filtered[COL_FAB] == filtro_fab]
    if filtro_sexo and filtro_sexo != "Todos":
        df_filtered = df_filtered[df_filtered[COL_SEXO] == filtro_sexo]
    df_filtered = df_filtered[(df_filtered[COL_IDADE] >= filtro_idade[0]) & (df_filtered[COL_IDADE] <= filtro_idade[1])]
    if filtro_categoria:
        df_filtered = df_filtered[df_filtered[categoria_col].isin(filtro_categoria)]

    st.info(f"**{len(df_filtered)}** registros encontrados com os filtros aplicados.")
    st.subheader("Tabela filtrada (Top 20)")
    st.dataframe(df_filtered.head(20), use_container_width=True)

    # --- Notícias/insights baseados no df_filtered ---
    st.markdown("---")
    st.subheader("Notícias e insights gerados")

    try:
        # usar o df_filtered para gerar insights contextuais
        if df_filtered.empty:
            st.warning("Sem registros no conjunto filtrado para gerar insights.")
        else:
            per_min = df_filtered[COL_DATA].min().strftime("%d/%m/%Y")
            per_max = df_filtered[COL_DATA].max().strftime("%d/%m/%Y")

            # por sexo
            fem_count = (df_filtered[COL_SEXO] == "FEMININO").sum()
            masc_count = (df_filtered[COL_SEXO] == "MASCULINO").sum()
            pct_fem = (fem_count / len(df_filtered) * 100) if len(df_filtered) else 0
            pct_masc = (masc_count / len(df_filtered) * 100) if len(df_filtered) else 0

            # fabricante e raça top (no conjunto filtrado)
            fab_top_filt = df_filtered[COL_FAB].mode().iloc[0] if not df_filtered[COL_FAB].mode().empty else "N/A"
            raca_top_filt = df_filtered[COL_RACA].mode().iloc[0] if not df_filtered[COL_RACA].mode().empty else "N/A"

            idade_media_filt = round(df_filtered[COL_IDADE].mean(), 1)

            # blocos de notícia (respeitando filtros)
            st.markdown(f"""
            <div class="news-block">
                <strong>Período:</strong> Entre {per_min} e {per_max}, o fabricante mais aplicado no recorte atual foi <strong>{fab_top_filt}</strong>.
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="news-block">
                <strong>Distribuição por sexo:</strong> No recorte atual, mulheres corresponderam a <strong>{pct_fem:.1f}%</strong> e homens a <strong>{pct_masc:.1f}%</strong> dos registros.
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="news-block">
                <strong>Idade média:</strong> A idade média no recorte é de <strong>{idade_media_filt}</strong> anos; a raça com maior frequência foi <strong>{raca_top_filt}</strong>.
            </div>
            """, unsafe_allow_html=True)

            # destaque temporal: período com maior número de vacinações (por mês) no recorte
            try:
                mensal = df_filtered.groupby(df_filtered[COL_DATA].dt.to_period("M")).size()
                top_mes = mensal.idxmax().strftime("%Y-%m")
                top_mes_val = int(mensal.max())
                st.markdown(f"""
                <div class="news-block">
                    <strong>Momento de pico:</strong> O mês com maior número de vacinações no recorte foi <strong>{top_mes}</strong> com <strong>{top_mes_val}</strong> registros.
                </div>
                """, unsafe_allow_html=True)
            except Exception:
                # se não houver dados de data (já tratamos), apenas ignore
                pass

    except Exception as e:
        st.warning("Não foi possível gerar insights automáticos. Erro: " + str(e))

elif module == "Tendências e Histórico":
    st.header("Tendências e Histórico")
    st.caption("Visualize distribuições e evolução temporal das vacinações.")

    col1, col2, col3 = st.columns(3)
    sexo_sel = col1.radio("Sexo", ["Todos"] + sorted(df[COL_SEXO].unique()))
    idade_sel = col2.slider("Faixa etária", int(df[COL_IDADE].min()), int(df[COL_IDADE].max()), (0, 100))
    raca_sel = col3.multiselect("Raça", sorted(df[COL_RACA].unique()), default=sorted(df[COL_RACA].unique()))

    df_hist = df.copy()
    if sexo_sel != "Todos":
        df_hist = df_hist[df_hist[COL_SEXO] == sexo_sel]
    df_hist = df_hist[(df_hist[COL_IDADE] >= idade_sel[0]) & (df_hist[COL_IDADE] <= idade_sel[1])]
    df_hist = df_hist[df_hist[COL_RACA].isin(raca_sel)]

    # Gráficos
    col1, col2 = st.columns(2)
    sexo_counts = df_hist[COL_SEXO].value_counts().reset_index()
    sexo_counts.columns = ['Sexo', 'Quantidade']
    fig_sexo = px.bar(sexo_counts, x='Sexo', y='Quantidade', color='Sexo', title="Distribuição por Sexo",
                      color_discrete_sequence=px.colors.qualitative.Safe)
    if sexo_sel == "Todos":
        col1.plotly_chart(fig_sexo, use_container_width=True)

    raca_counts = df_hist[COL_RACA].value_counts().reset_index()
    raca_counts.columns = ['Raça', 'Quantidade']
    fig_raca = px.pie(raca_counts, names='Raça', values='Quantidade', hole=0.3, title="Distribuição por Raça",
                      color_discrete_sequence=px.colors.qualitative.Safe)
    col2.plotly_chart(fig_raca, use_container_width=True)

    fab_counts = df_hist[COL_FAB].value_counts().reset_index()
    fab_counts.columns = ['Fabricante', 'Quantidade']
    fig_fab = px.pie(fab_counts, names='Fabricante', values='Quantidade', hole=0.3, title="Distribuição por Fabricante",
                     color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig_fab, use_container_width=True)

    # Vacinações por idade
    idade_hist = df_hist.groupby(COL_IDADE).size().reset_index(name='Quantidade')
    fig_idade = px.line(idade_hist, x=COL_IDADE, y='Quantidade', markers=True, title="Vacinações por Idade")
    st.plotly_chart(fig_idade, use_container_width=True)

elif module == "Agrupamentos Inteligentes":
    st.header("Agrupamentos Inteligentes (K-Means)")
    st.caption("Agrupe perfis semelhantes de vacinados. O agrupamento é automático e a qualidade é apresentada.")

    n_clusters = st.slider("Número de grupos", min_value=2, max_value=8, value=3)

    # preparar dados: dummies para colunas categóricas principais
    df_ml = df[[COL_IDADE, COL_SEXO, COL_RACA, COL_FAB]].copy()
    # converter para versões sem acento/normalizadas onde faz sentido
    df_ml[COL_SEXO] = df_ml[COL_SEXO].map(lambda x: strip_accents(x).upper())
    df_ml[COL_RACA] = df_ml[COL_RACA].map(lambda x: strip_accents(x).upper())
    df_ml = pd.get_dummies(df_ml, columns=[COL_SEXO, COL_RACA, COL_FAB], drop_first=True)

    X = df_ml.select_dtypes(include=np.number).fillna(0)

    if st.button("Gerar agrupamento"):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        score = silhouette_score(X, clusters) if len(X) > n_clusters else float("nan")

        df_clustered = df.copy()
        df_clustered["Grupo"] = clusters

        st.success(f"Agrupamento realizado. Qualidade dos grupos (silhueta): {score:.3f}" if not np.isnan(score) else "Agrupamento realizado.")
        profile = df_clustered.groupby("Grupo").agg(
            Idade_Media=(COL_IDADE, "mean"),
            Total=(COL_IDADE, "count")
        ).round(2)
        st.subheader("Resumo dos grupos")
        st.dataframe(profile, use_container_width=True)

        # PCA para visualização 2D
        pca = PCA(n_components=2)
        comp = pca.fit_transform(X)
        df_clustered["Eixo_1"] = comp[:, 0]
        df_clustered["Eixo_2"] = comp[:, 1]

        fig = px.scatter(df_clustered, x="Eixo_1", y="Eixo_2", color=df_clustered["Grupo"].astype(str),
                         title="Visualização 2D dos Grupos")
        st.plotly_chart(fig, use_container_width=True)

elif module == "Projeções Futuras":
    st.header("Projeções Futuras de Vacinação")
    st.caption("Projeções simples com regressão linear. Ajuste o horizonte e filtros por sexo se necessário.")

    col1, col2 = st.columns(2)
    horizonte = col1.slider("Meses à frente", 1, 36, 12)
    sexo_proj = col2.multiselect("Filtrar por sexo", ["Todos"] + sorted(df[COL_SEXO].unique()), default=["Todos"])

    df_pred = df.copy()
    if "Todos" not in sexo_proj:
        df_pred = df_pred[df_pred[COL_SEXO].isin(sexo_proj)]

    # agrupar por mês (período)
    df_mes = df_pred.groupby(df_pred[COL_DATA].dt.to_period("M")).size().reset_index(name="Vacinações")
    df_mes[COL_DATA] = df_mes[df_mes.columns[0]]  # usar a coluna período como referência
    df_mes["Indice"] = np.arange(len(df_mes))

    X = df_mes[["Indice"]]
    y = df_mes["Vacinações"]

    if len(X) < 2:
        st.warning("Dados insuficientes para projeção. São necessários pelo menos 2 pontos mensais.")
    else:
        modelo = LinearRegression()
        modelo.fit(X, y)

        X_future = np.arange(len(X), len(X) + horizonte).reshape(-1, 1)
        y_future = modelo.predict(X_future)

        futuras_datas = pd.period_range(df_mes[COL_DATA].iloc[-1] + 1, periods=horizonte, freq="M")
        df_proj = pd.DataFrame({"Mês": futuras_datas.astype(str), "Vacinações Previstas": y_future})

        df_real = pd.DataFrame({
            "Mês": df_mes[COL_DATA].astype(str),
            "Vacinações": df_mes["Vacinações"]
        })

        df_all = pd.concat([
            df_real.assign(Tipo="Real").rename(columns={"Vacinações": "Valor"}),
            df_proj.assign(Tipo="Previsto").rename(columns={"Vacinações Previstas": "Valor"})
        ], ignore_index=True)

        fig = px.line(df_all, x="Mês", y="Valor", color="Tipo", markers=True, title="Real vs Previsto")
        st.plotly_chart(fig, use_container_width=True)

        tendencia_text = "Crescimento" if y_future[-1] > y.iloc[-1] else "Queda"
        st.metric("Tendência atual", tendencia_text)