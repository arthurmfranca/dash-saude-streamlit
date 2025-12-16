import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")

def get_csv_download_link(df):
    csv = df.to_csv(index=False).encode("utf-8")
    return csv

st.set_page_config(page_title="Dashboard Dummy", layout="wide")

st.sidebar.header("Fonte de dados")

arquivo_fato = st.sidebar.text_input(
    "Arquivo Fato CSV (pasta data/)",
    value="adesao.csv"
)

arquivo_dim = st.sidebar.text_input(
    "Arquivo Dimensão CSV (opcional, pasta data/)",
    value="municipios.csv"
)

# ===== BOTÃO CARREGAR ARQUI () =====
if st.sidebar.button("Carregar dados", key="btn_carregar"):
    try:
        # Carrega tabela fato (obrigatória)
        df_fato = pd.read_csv(
            f"data/{arquivo_fato}",
            encoding="utf-8",
            sep=",",
        )

        # Tabela dimensão é opcional
        df_dim = None
        if arquivo_dim.strip():  # só tenta ler se não estiver vazio
            try:
                df_dim = pd.read_csv(
                    f"data/{arquivo_dim}",
                    encoding="utf-8",
                    sep=",",
                )
            except FileNotFoundError:
                st.sidebar.warning(f"Arquivo de dimensão não encontrado: {arquivo_dim}. Seguindo só com a fato.")
                df_dim = None

        # Tratamento de tipos na fato
        for col in ["Valor Adesão", "Código IBGE", "Código Macro"]:
            if col in df_fato.columns:
                df_fato[col] = pd.to_numeric(df_fato[col], errors="coerce")

        for col in df_fato.columns:
            if "data" in col.lower():
                df_fato[col] = pd.to_datetime(df_fato[col], dayfirst=True, errors="coerce")

        st.session_state.df_fato = df_fato
        st.session_state.df_dim = df_dim
        st.sidebar.success("Dados carregados!")
    except Exception as e:
        st.sidebar.error(f"Erro ao ler arquivos: {e}")
# ===== FIM DO BLOCO DO BOTÃO =====

# =====================================================
# BLOCO PRINCIPAL APÓS CARREGAR O CSV
# =====================================================
if "df_fato" in st.session_state:
    df_fato = st.session_state.df_fato.copy()
    df_dim = st.session_state.df_dim

    st.header("Modelo de dados (join)")

    if df_dim is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            chave_fato = st.selectbox(
                "Chave na tabela Fato",
                df_fato.columns,
                index=list(df_fato.columns).index("Código IBGE") if "Código IBGE" in df_fato.columns else 0,
                key="chave_fato"
            )
        with col2:
            chave_dim = st.selectbox(
                "Chave na tabela Dimensão",
                df_dim.columns,
                index=list(df_dim.columns).index("Código IBGE") if "Código IBGE" in df_dim.columns else 0,
                key="chave_dim"
            )
        with col3:
            tipo_join = st.selectbox(
                "Tipo de junção",
                ["left", "inner"],
                key="tipo_join"
            )

        df_modelo = df_fato.merge(
            df_dim,
            left_on=chave_fato,
            right_on=chave_dim,
            how=tipo_join,
            suffixes=("_fato", "_dim")
        )
    else:
        st.info("Nenhuma tabela dimensão carregada. Usando apenas tabela fato.")
        df_modelo = df_fato

    # A partir daqui usamos df_modelo como base para filtros/medidas
    df_original = df_modelo.copy()
    st.sidebar.subheader("Filtros")
    df = df_original.copy()


# UF
    if "UF" in df.columns:
        opcoes_uf = sorted(df["UF"].dropna().unique())
        uf_sel = st.sidebar.multiselect("UF", opcoes_uf, default=opcoes_uf)
        if uf_sel:  # só filtra se houver seleção
            df = df[df["UF"].isin(uf_sel)]

# Município
    if "Município" in df.columns:
        opcoes_mun = sorted(df["Município"].dropna().unique())
        mun_sel = st.sidebar.multiselect("Município", opcoes_mun)
        if mun_sel:
            df = df[df["Município"].isin(mun_sel)]

# Adesão
    if "Adesão" in df.columns:
        opcoes_adesao = sorted(df["Adesão"].dropna().unique())
        adesao_sel = st.sidebar.multiselect("Adesão", opcoes_adesao, default=opcoes_adesao)
        if adesao_sel:
            df = df[df["Adesão"].isin(adesao_sel)]

# Macrorregião
    if "Macrorregião de Saúde" in df.columns:
        opcoes_macro = sorted(df["Macrorregião de Saúde"].dropna().unique())
        macro_sel = st.sidebar.multiselect("Macrorregião de Saúde", opcoes_macro)
        if macro_sel:
            df = df[df["Macrorregião de Saúde"].isin(macro_sel)]

# Data
    if "Data" in df.columns and str(df["Data"].dtype).startswith("datetime"):
        if not df["Data"].isna().all():
            data_min = df["Data"].min()
            data_max = df["Data"].max()
            data_inicio, data_fim = st.sidebar.date_input(
                "Intervalo de Data",
                value=(data_min.date(), data_max.date())
            )
            # garante que veio uma tupla com 2 datas
            if isinstance(data_inicio, type(data_max.date())) and isinstance(data_fim, type(data_max.date())):
                df = df[
                    (df["Data"] >= pd.to_datetime(data_inicio)) &
                    (df["Data"] <= pd.to_datetime(data_fim))
                ]


    # -------------------------
    # PRÉ-VISUALIZAÇÃO JÁ FILTRADA
    # -------------------------
    st.write("Tipos das colunas (após conversão):")
    st.write(df.dtypes)

    st.header("Pré-visualização dos dados filtrados")
    st.dataframe(df.head())

    # -------------------------
    # EXPORTAR DADOS FILTRADOS PARA CSV
    # -------------------------
    st.subheader("Exportar dados")

    csv_bytes = convert_df(df)

    st.download_button(
        label="Baixar dados filtrados em CSV",
        data=csv_bytes,
        file_name="dados_filtrados.csv",
        mime="text/csv",
        key="download_csv_filtrado"
    )


    # -------------------------
    # MEDIDAS (usando df FILTRADO)
    # -------------------------
    st.header("Medidas (tipo DAX)")

    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(exclude=["number", "datetime64[ns]"]).columns

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        medida_tipo = st.selectbox(
            "Tipo de medida",
            [
                "SUM",
                "AVG",
                "COUNT",
                "DISTINCT COUNT",
                "% do total por categoria",
                "TOP N",
                "Taxa (num/denom) x K"
            ],
            key="tipo_medida"
        )
    
    with col_m2:
        medida_coluna = st.selectbox("Coluna numérica", num_cols, key="coluna_medida")
    with col_m3:
        medida_nome = st.text_input("Nome da medida", value="Medida_1", key="nome_medida")

    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(exclude=["number", "datetime64[ns]"]).columns

    cat_group = None
    top_n = None
    num_col_taxa = None
    den_col_taxa = None
    k_const = None

    if medida_tipo == "% do total por categoria":
        cat_group = st.selectbox("Agrupar por (categoria)", cat_cols, key="cat_group")

    if medida_tipo == "TOP N":
        top_n = st.number_input("Valor de N (Top N)", min_value=1, value=10, key="top_n_val")

    if medida_tipo == "Taxa (num/denom) x K":
        num_col_taxa = st.selectbox("Coluna numerador", num_cols, key="num_col_taxa")
        den_col_taxa = st.selectbox("Coluna denominador", num_cols, key="den_col_taxa")
        k_const = st.number_input("Constante K (ex.: 1000, 100000)", min_value=1.0, value=100000.0, key="k_const")

    if st.button("Calcular medida", key="btn_calc_medida"):
        df_result = None

        if medida_tipo == "SUM":
            valor = df[medida_coluna].sum()
            df_result = pd.DataFrame({medida_nome: [valor]})

        elif medida_tipo == "AVG":
            valor = df[medida_coluna].mean()
            df_result = pd.DataFrame({medida_nome: [valor]})

        elif medida_tipo == "COUNT":
            valor = df[medida_coluna].count()
            df_result = pd.DataFrame({medida_nome: [valor]})

        elif medida_tipo == "DISTINCT COUNT":
            valor = df[medida_coluna].nunique()
            df_result = pd.DataFrame({medida_nome: [valor]})

        elif medida_tipo == "% do total por categoria" and cat_group is not None:
            agrupado = df.groupby(cat_group, as_index=False)[medida_coluna].sum()
            total = agrupado[medida_coluna].sum()
            agrupado[medida_nome] = agrupado[medida_coluna] / total * 100
            df_result = agrupado

        elif medida_tipo == "TOP N" and top_n is not None:
            df_result = df.nlargest(top_n, medida_coluna)

        elif medida_tipo == "Taxa (num/denom) x K" and num_col_taxa and den_col_taxa and k_const:
        # soma numerador e denominador (agregação global)
            num = df[num_col_taxa].sum()
            den = df[den_col_taxa].sum()

            if den == 0 or pd.isna(den):
                taxa = None
            else:
                taxa = num / den * k_const

            df_result = pd.DataFrame({
                "Numerador": [num],
                "Denominador": [den],
                f"Taxa_{medida_nome}": [taxa],
                "K": [k_const]
            })

        if df_result is not None:
            st.subheader("Resultado da medida")
            st.dataframe(df_result)

            st.session_state.df_medida = df_result

            st.subheader("Exportar resultado da medida")

            if "df_medida" in st.session_state:
                csv_medida = convert_df(st.session_state.df_medida)

                st.download_button(
                    label="Baixar resultado da medida (CSV)",
                    data=csv_medida,
                    file_name="resultado_medida.csv",
                    mime="text/csv",
                    key="download_medida_csv"
                )

            else:
                st.info("Nenhuma medida calculada ainda.")    

    # -------------------------
    # GRÁFICO RÁPIDO (df filtrado)
    # -------------------------
    st.header("Gráfico rápido")

    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) == 0:
        st.info("Não há colunas numéricas para gráfico.")
    else:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            x_col = st.selectbox("Eixo X", df.columns, key="x_col")
        with col_g2:
            y_col = st.selectbox("Eixo Y (numérico)", num_cols, key="y_col")

        tipo_graf = st.selectbox(
            "Tipo de gráfico",
            ["Barra", "Linha", "Pizza", "Scatter"],
            key="tipo_grafico"
        )


        if st.button("Gerar gráfico", key="btn_grafico"):
            if tipo_graf == "Barra":
                fig = px.bar(df, x=x_col, y=y_col)
            elif tipo_graf == "Linha":
                fig = px.line(df, x=x_col, y=y_col)
            elif tipo_graf == "Pizza":
                fig = px.pie(df, names=x_col, values=y_col)
            elif tipo_graf == "Scatter":
                fig = px.scatter(df, x=x_col, y=y_col)

            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Informe o nome do CSV na barra lateral e clique em 'Carregar CSV'.")
