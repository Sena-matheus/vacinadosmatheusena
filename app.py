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
            return cols_norm[cand_norm]
    return None

def safe_series_str_normalize(s: pd.Series):
    return s.astype(str).map(lambda x: strip_accents(x).upper().strip())

st.set_page_config(page_title="Painel de Vacinação", layout="wide")
st.title("Painel Interativo de Vacinação")
st.markdown("Visualize dados, tendências e previsões de vacinação de forma interativa e profissional.")

# CSS customizado
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    h1, h2, h3 { color: #0b5a4a; }
    .news-block {
        background-color: #f7faf9;
        border-left: 4px solid #0b5a4a;
        padding: 10px 15px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .stHeader { padding-bottom: 8px; }
    .main .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file_path: str):
    """
    Tenta usar o helper load_and_preprocess_data; fallback a leitura direta com diferentes encodings.
    Normaliza tipos e retorna (df, cols_detectadas).
    """
    try:
        df = load_and_preprocess_data(file_path)
    except Exception:
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except Exception:
            df = pd.read_csv(file_path, encoding="latin-1", sep=None, engine="python")

    df.columns = [c.strip() for c in df.columns]

    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).map(lambda x: x.strip())

    col_data = find_col(df, ["Data da Vacinação", "Data da Vacinacao", "data_da_vacinacao", "data_vacinacao", "data"])
    col_sexo = find_col(df, ["Sexo", "sexo"])
    col_raca = find_col(df, ["Raça", "Raca", "raca"])
    col_idade = find_col(df, ["Idade", "idade", "idade_anos", "idade_anos_int"])
    col_fab = find_col(df, ["Fabricante da Vacina", "Fabricante", "fabricante_da_vacina", "fabricante"])

    if col_data:
        df[col_data] = pd.to_datetime(df[col_data], errors="coerce")
        df = df.dropna(subset=[col_data])

    if col_sexo:
        df[col_sexo] = safe_series_str_normalize(df[col_sexo])
    if col_raca:
        df[col_raca] = safe_series_str_normalize(df[col_raca])
    if col_fab:
        df[col_fab] = df[col_fab].astype(str).map(lambda x: x.strip())
    if col_idade:
        df[col_idade] = pd.to_numeric(df[col_idade], errors="coerce").fillna(0).astype(int)

    return df, {
        "data": col_data,
        "sexo": col_sexo,
        "raca": col_raca,
        "idade": col_idade,
        "fabricante": col_fab
    }

FILE_PATH = "vacinados.csv"
df, COLS = load_data(FILE_PATH)

essential = ["data", "sexo", "raca", "idade", "fabricante"]
missing = [k for k in essential if COLS.get(k) is None]
if missing:
    st.error(f"Colunas essenciais não encontradas no arquivo: {missing}. Verifique os nomes do CSV.")
    st.stop()

COL_DATA = COLS["data"]
COL_SEXO = COLS["sexo"]
COL_RACA = COLS["raca"]
COL_IDADE = COLS["idade"]
COL_FAB = COLS["fabricante"]

df["_mes_periodo"] = df[COL_DATA].dt.to_period("M")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de registros", f"{len(df):,}".replace(",", "."))
col2.metric("Idade média", round(df[COL_IDADE].mean(), 1))
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
    st.caption("Use os filtros para refinar a visualização e explorar insights interativos.")

    col1, col2 = st.columns(2)
    filtro_raca = col1.selectbox("Raça", ["Todas"] + sorted(df[COL_RACA].unique()))
    filtro_fab = col2.selectbox("Fabricante", ["Todos"] + sorted(df[COL_FAB].unique()))

    col3, col4, col5 = st.columns(3)
    filtro_sexo = col3.radio("Sexo", ["Todos"] + sorted(df[COL_SEXO].unique()))
    filtro_idade = col4.slider("Faixa etária", int(df[COL_IDADE].min()), int(df[COL_IDADE].max()), (0, 100))

    data_min = df[COL_DATA].min()
    data_max = df[COL_DATA].max()
    filtro_periodo = col5.slider(
        "Período de vacinação",
        min_value=data_min.to_pydatetime(),
        max_value=data_max.to_pydatetime(),
        value=(data_min.to_pydatetime(), data_max.to_pydatetime()),
        format="DD/MM/YYYY"
    )

    categoria_col = find_col(df, ["Categoria", "categoria"])
    categorias = sorted(df[categoria_col].unique()) if categoria_col else []
    filtro_categoria = st.multiselect("Categoria", categorias) if categorias else []

    df_filtered = df.copy()
    if filtro_raca != "Todas":
        df_filtered = df_filtered[df_filtered[COL_RACA] == filtro_raca]
    if filtro_fab != "Todos":
        df_filtered = df_filtered[df_filtered[COL_FAB] == filtro_fab]
    if filtro_sexo != "Todos":
        df_filtered = df_filtered[df_filtered[COL_SEXO] == filtro_sexo]
    df_filtered = df_filtered[
        (df_filtered[COL_IDADE] >= filtro_idade[0]) & (df_filtered[COL_IDADE] <= filtro_idade[1])
    ]
    if filtro_categoria:
        df_filtered = df_filtered[df_filtered[categoria_col].isin(filtro_categoria)]
    df_filtered = df_filtered[
        (df_filtered[COL_DATA] >= filtro_periodo[0]) & (df_filtered[COL_DATA] <= filtro_periodo[1])
    ]

    st.info(f"**{len(df_filtered)}** registros encontrados com os filtros aplicados.")
    st.dataframe(df_filtered.head(20), use_container_width=True)

elif module == "Tendências e Histórico":
    st.header("Tendências e Histórico")
    st.caption("Visualize distribuições e evolução temporal das vacinações, com insights detalhados.")

    col1, col2, col3 = st.columns(3)
    sexo_sel = col1.radio("Sexo", ["Todos"] + sorted(df[COL_SEXO].unique()))
    idade_sel = col2.slider("Faixa etária", int(df[COL_IDADE].min()), int(df[COL_IDADE].max()), (0, 100))
    raca_sel = col3.multiselect("Raça", sorted(df[COL_RACA].unique()), default=sorted(df[COL_RACA].unique()))

    data_min = df[COL_DATA].min()
    data_max = df[COL_DATA].max()
    periodo_sel = st.slider(
        "Período de vacinação",
        min_value=data_min.to_pydatetime(),
        max_value=data_max.to_pydatetime(),
        value=(data_min.to_pydatetime(), data_max.to_pydatetime()),
        format="DD/MM/YYYY"
    )

    df_hist = df.copy()
    if sexo_sel != "Todos":
        df_hist = df_hist[df_hist[COL_SEXO] == sexo_sel]
    df_hist = df_hist[(df_hist[COL_IDADE] >= idade_sel[0]) & (df_hist[COL_IDADE] <= idade_sel[1])]
    df_hist = df_hist[df_hist[COL_RACA].isin(raca_sel)]
    df_hist = df_hist[(df_hist[COL_DATA] >= periodo_sel[0]) & (df_hist[COL_DATA] <= periodo_sel[1])]

    col1, col2 = st.columns(2)
    sexo_counts = df_hist[COL_SEXO].value_counts().reset_index()
    sexo_counts.columns = ['Sexo', 'Quantidade']
    fig_sexo = px.bar(sexo_counts, x='Sexo', y='Quantidade', color='Sexo', title="Distribuição por Sexo")
    if sexo_sel == "Todos":
        col1.plotly_chart(fig_sexo, use_container_width=True)

    raca_counts = df_hist[COL_RACA].value_counts().reset_index()
    raca_counts.columns = ['Raça', 'Quantidade']
    fig_raca = px.pie(raca_counts, names='Raça', values='Quantidade', hole=0.3, title="Distribuição por Raça")
    col2.plotly_chart(fig_raca, use_container_width=True)

    fab_counts = df_hist[COL_FAB].value_counts().reset_index()
    fab_counts.columns = ['Fabricante', 'Quantidade']
    fig_fab = px.pie(fab_counts, names='Fabricante', values='Quantidade', hole=0.3, title="Distribuição por Fabricante")
    st.plotly_chart(fig_fab, use_container_width=True)

    idade_hist = df_hist.groupby(COL_IDADE).size().reset_index(name='Quantidade')
    fig_idade = px.line(idade_hist, x=COL_IDADE, y='Quantidade', markers=True, title="Vacinações por Idade")
    st.plotly_chart(fig_idade, use_container_width=True)

    st.markdown("---")
    st.subheader("Notícias e insights históricos")

    if df_hist.empty:
        st.warning("Sem registros no conjunto filtrado para gerar insights.")
    else:
        per_min = df_hist[COL_DATA].min().strftime("%d/%m/%Y")
        per_max = df_hist[COL_DATA].max().strftime("%d/%m/%Y")

        fem_count = (df_hist[COL_SEXO] == "FEMININO").sum()
        masc_count = (df_hist[COL_SEXO] == "MASCULINO").sum()
        pct_fem = (fem_count / len(df_hist) * 100) if len(df_hist) else 0.0
        pct_masc = (masc_count / len(df_hist) * 100) if len(df_hist) else 0.0

        fab_top_filt = df_hist[COL_FAB].mode().iloc[0] if not df_hist[COL_FAB].mode().empty else "N/A"
        raca_top_filt = df_hist[COL_RACA].mode().iloc[0] if not df_hist[COL_RACA].mode().empty else "N/A"
        idade_media_filt = round(df_hist[COL_IDADE].mean(), 1)

        mensal = df_hist.groupby(df_hist[COL_DATA].dt.to_period("M")).size()
        if not mensal.empty:
            top_mes = mensal.idxmax().strftime("%Y-%m")
            top_mes_val = int(mensal.max())
        else:
            top_mes, top_mes_val = "N/A", 0

        with st.expander("Período analisado"):
            st.markdown(f"""
            <div class="news-block">
                Entre <strong>{per_min}</strong> e <strong>{per_max}</strong>, o fabricante mais aplicado foi <strong>{fab_top_filt}</strong>.
            </div>
            """, unsafe_allow_html=True)

            df_mensal = df_hist.groupby(df_hist[COL_DATA].dt.to_period("M")).size().reset_index(name="Vacinações")
            period_col = df_mensal.columns[0]
            df_mensal["Periodo"] = df_mensal[period_col].astype(str)
            fig = px.line(df_mensal, x="Periodo", y="Vacinações", markers=True, title="Evolução das vacinações no período")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Distribuição por sexo"):
            st.markdown(f"""
            <div class="news-block">
                Mulheres: <strong>{pct_fem:.1f}%</strong>, Homens: <strong>{pct_masc:.1f}%</strong> dos registros filtrados.
            </div>
            """, unsafe_allow_html=True)

            sexo_counts = df_hist[COL_SEXO].value_counts().reset_index()
            sexo_counts.columns = ["Sexo", "Quantidade"]
            fig = px.bar(sexo_counts, x="Sexo", y="Quantidade", color="Sexo", title="Distribuição por sexo")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Faixa etária média"):
            st.markdown(f"""
            <div class="news-block">
                Idade média: <strong>{idade_media_filt} anos</strong>. Raça predominante: <strong>{raca_top_filt}</strong>.
            </div>
            """, unsafe_allow_html=True)

            idade_hist = df_hist.groupby(COL_IDADE).size().reset_index(name="Quantidade")
            fig = px.histogram(idade_hist, x=COL_IDADE, y="Quantidade", nbins=20, title="Distribuição por faixa etária")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Pico de vacinação"):
            st.markdown(f"""
            <div class="news-block">
                O mês com mais vacinações foi <strong>{top_mes}</strong>, com <strong>{top_mes_val} registros</strong>.
            </div>
            """, unsafe_allow_html=True)

            mensal_df = mensal.reset_index()
            mensal_df.columns = ["Mês", "Quantidade"]
            mensal_df["Mês"] = mensal_df["Mês"].astype(str)
            fig = px.bar(mensal_df, x="Mês", y="Quantidade", title="Vacinações mensais")
            st.plotly_chart(fig, use_container_width=True)

if module == "Agrupamentos Inteligentes":
    st.header("Agrupamentos Inteligentes (K-Means)")
    st.markdown("""
    <div class="news-block">
        Nesta seção, aplicamos técnicas de aprendizado não supervisionado para **agrupar registros** com perfis semelhantes.
        O método K-Means permite identificar padrões ocultos e gerar insights estratégicos.
    </div>
    """, unsafe_allow_html=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("É necessário pelo menos duas variáveis numéricas para realizar o agrupamento.")
    else:
        selected_vars = st.multiselect(
            "Selecione as variáveis para agrupamento:",
            numeric_cols,
            default=numeric_cols[:3]
        )
        n_clusters = st.slider("Número de clusters:", 2, 10, 3)

        if st.button("Gerar Clusters"):
            X = df[selected_vars].dropna()
            if len(X) < n_clusters:
                st.warning("O número de amostras é insuficiente para formar os clusters escolhidos.")
            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X)
                df['Cluster'] = clusters

                silhouette = silhouette_score(X, clusters)
                st.markdown(f"""
                <div class="news-block">
                    <strong>Coeficiente de Silhueta:</strong> {silhouette:.3f}<br>
                    Quanto mais próximo de 1, melhor a separação entre grupos.
                </div>
                """, unsafe_allow_html=True)

                st.dataframe(df[['Cluster'] + selected_vars].head(10))

                if X.shape[0] > 1 and X.shape[1] > 1:
                    n_components = min(2, X.shape[1])
                    pca = PCA(n_components=n_components)
                    try:
                        comp = pca.fit_transform(X)
                        df_plot = pd.DataFrame({
                            'Componente 1': comp[:, 0],
                            'Componente 2': comp[:, 1] if n_components > 1 else 0,
                            'Cluster': clusters
                        })
                        fig_pca = px.scatter(
                            df_plot,
                            x='Componente 1',
                            y='Componente 2',
                            color='Cluster',
                            title='Clusters de registros (PCA)'
                        )
                        st.plotly_chart(fig_pca, use_container_width=True)
                    except ValueError as e:
                        st.warning(f"PCA não pôde ser aplicado: {e}")
                else:
                    st.warning("Não há dados suficientes para aplicar o PCA.")

elif module == "Projeções Futuras":
    st.header("Projeções Futuras de Vacinação")
    st.caption("Projeções simples com regressão linear. Ajuste o horizonte e filtros por sexo se necessário.")

    col1, col2 = st.columns(2)
    horizonte = col1.slider("Meses à frente", 1, 36, 12)
    sexo_proj = col2.multiselect("Filtrar por sexo", ["Todos"] + sorted(df[COL_SEXO].unique()), default=["Todos"])

    df_pred = df.copy()
    if "Todos" not in sexo_proj:
        df_pred = df_pred[df_pred[COL_SEXO].isin(sexo_proj)]

    df_mes = df_pred.groupby(df_pred[COL_DATA].dt.to_period("M")).size().reset_index(name="Vacinações")
    period_col = df_mes.columns[0]
    df_mes["Periodo"] = df_mes[period_col].astype(str)
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

        futuras_datas = pd.period_range(df_mes[period_col].iloc[-1] + 1, periods=horizonte, freq="M")
        df_proj = pd.DataFrame({"Mês": futuras_datas.astype(str), "Vacinações Previstas": y_future})

        df_real = pd.DataFrame({
            "Mês": df_mes["Periodo"],
            "Vacinações": df_mes["Vacinações"]
        })

        df_all = pd.concat([
            df_real.assign(Tipo="Real").rename(columns={"Vacinações": "Valor"}),
            df_proj.assign(Tipo="Previsto").rename(columns={"Vacinações Previstas": "Valor"})
        ], ignore_index=True)

        fig = px.line(df_all, x="Mês", y="Valor", color="Tipo", markers=True, title="Real vs Previsto")
        st.plotly_chart(fig, use_container_width=True)

        tendencia = "aumento" if y_future[-1] > y.iloc[-1] else "queda"
        st.markdown(f"""
        <div class="news-block">
            <strong>Tendência:</strong> projeção indica <strong>{tendencia}</strong> nas vacinações nos próximos {horizonte} meses.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Projeção por Sexo")

        sexos_disponiveis = df[COL_SEXO].dropna().unique()
        proj_por_sexo = []

        for sexo in sexos_disponiveis:
            df_sexo = df_pred[df_pred[COL_SEXO] == sexo]
            df_mes_sexo = df_sexo.groupby(df_sexo[COL_DATA].dt.to_period("M")).size().reset_index(name="Vacinações")
            df_mes_sexo["Indice"] = np.arange(len(df_mes_sexo))

            if len(df_mes_sexo) >= 2:
                Xs = df_mes_sexo[["Indice"]]
                ys = df_mes_sexo["Vacinações"]
                modelo_s = LinearRegression().fit(Xs, ys)

                Xs_future = np.arange(len(Xs), len(Xs) + horizonte).reshape(-1, 1)
                ys_future = modelo_s.predict(Xs_future)

                futuras_datas_s = pd.period_range(df_mes_sexo.iloc[-1, 0] + 1, periods=horizonte, freq="M")
                df_proj_s = pd.DataFrame({
                    "Mês": futuras_datas_s.astype(str),
                    "Vacinações Previstas": ys_future,
                    "Sexo": sexo
                })

                df_real_s = pd.DataFrame({
                    "Mês": df_mes_sexo.iloc[:, 0].astype(str),
                    "Vacinações": ys,
                    "Sexo": sexo
                })

                df_comb = pd.concat([
                    df_real_s.assign(Tipo="Real", Valor=df_real_s["Vacinações"]),
                    df_proj_s.assign(Tipo="Previsto", Valor=df_proj_s["Vacinações Previstas"])
                ], ignore_index=True)

                proj_por_sexo.append(df_comb)

        if proj_por_sexo:
            df_total_proj = pd.concat(proj_por_sexo, ignore_index=True)
            fig_sexo_proj = px.line(
                df_total_proj,
                x="Mês",
                y="Valor",
                color="Sexo",
                line_dash="Tipo",
                markers=True,
                title="Projeção de vacinações futuras por sexo"
            )
            st.plotly_chart(fig_sexo_proj, use_container_width=True)

            ultimos = df_total_proj[df_total_proj["Tipo"] == "Previsto"].groupby("Sexo")["Valor"].sum()
            if len(ultimos) >= 1:
                sexo_destaque = ultimos.idxmax()
                st.markdown(f"""
                <div class="news-block">
                    Projeções indicam que <strong>{sexo_destaque}</strong> deverá apresentar maior volume de vacinações
                    nos próximos <strong>{horizonte} meses</strong>.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Dados insuficientes para calcular projeção por sexo.")
