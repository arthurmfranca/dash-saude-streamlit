"""
🧮 FERRAMENTA DE ANÁLISE ESTATÍSTICA AVANÇADA
Autor: Arthur (Estudos Ciência de Dados)
Objetivo: Análise descritiva + testes inferenciais + ML com interpretação automática
Skills: Estatística aplicada, Streamlit, Pandas, SciPy, Scikit-learn
"""

<<<<<<< HEAD
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
=======
import streamlit as st  # Interface web interativa
import pandas as pd     # Manipulação de dados
import numpy as np      # Cálculos numéricos
import plotly.express as px      # Gráficos interativos
import plotly.graph_objects as go # Gráficos avançados
from plotly.subplots import make_subplots  # Subplots múltiplos
import scipy.stats as stats       # TESTES ESTATÍSTICOS (t-test, ANOVA, etc)
from sklearn.preprocessing import StandardScaler  # Padronização ML
from sklearn.decomposition import PCA            # Análise de componentes
from sklearn.cluster import KMeans               # Clustering
import warnings
warnings.filterwarnings('ignore')  # Remove warnings desnecessários

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA (EXECUTA 1x por sessão)
# =============================================================================
st.set_page_config(
    page_title="🧮 Estatística Avançada - Estudos", 
    layout="wide",  # Layout largo (melhor para dashboards)
    initial_sidebar_state="expanded"  # Sidebar sempre aberta
)

# Função utilitária: Export CSV (cache para performance)
@st.cache_data  # 🚀 CACHE: executa 1x, reutiliza resultado
def convert_df(df):
    """Converte DataFrame para bytes CSV (download)"""
    return df.to_csv(index=False).encode("utf-8")

# =============================================================================
# INTERFACE PRINCIPAL
# =============================================================================
st.title("🧮 Ferramenta de Análise Estatística Avançada")
st.markdown("""
**Para seus estudos de estatística aplicada e machine learning**
- 📊 Análise descritiva completa
- 🔍 Correlação + testes inferenciais  
- 🤖 Clustering + PCA
- 💡 Interpretação automática em português
""")

# =====================================
# PASSO 1: UPLOAD DE DADOS
# =====================================
uploaded_file = st.file_uploader(
    "📁 Carregue sua base de dados (CSV/Excel)", 
    type=['csv','xlsx'],
    help="Qualquer dataset numérico/categórico"
)

if uploaded_file is not None:
    # Detecta formato automaticamente
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    else:
        df = pd.read_excel(uploaded_file)
    
    # ✅ CONFIRMAÇÃO VISUAL
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Linhas", f"{df.shape[0]:,}")
    col2.metric("Colunas", df.shape[1])
    col3.metric("Numéricas", len(df.select_dtypes(include=[np.number]).columns))
    col4.metric("Categóricas", len(df.select_dtypes(exclude=[np.number]).columns))
    
    st.dataframe(df.head(10), use_container_width=True)

    # =====================================
    # 🔍 DIAGNÓSTICO AUTOMÁTICO
    # =====================================
    st.subheader("🔍 Diagnóstico Automático do Dataset")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**📋 Todas as colunas:**")
        for i, col in enumerate(df.columns):
            st.write(f"{i+1}. **{col}** ({df[col].dtype})")
    with col2:
        st.write("**🔢 Colunas Numéricas:**")
        num_cols_diag = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols_diag:
            st.write(f"• {col}")
    with col3:
        st.write("**🏷️ Colunas Categóricas:**")
        cat_cols_diag = df.select_dtypes(exclude=[np.number]).columns.tolist()
        for col in cat_cols_diag:
            st.write(f"• {col}")
    # FIM DIAGNÓSTICO
    
    # =====================================
    # PASSO 2: MENU LATERAL - SELEÇÃO DE ANÁLISES
    # =====================================
    st.sidebar.header("🔍 Escolha sua Análise")
    analise_tipo = st.sidebar.selectbox("Tipo de análise", [
    "📊 1. Análise Descritiva", 
    "🔗 2. Correlação", 
    "🏥 3. Análise Saúde",           # ← ADICIONE ESTA LINHA
    "📈 4. Testes Estatísticos", 
    "🎭 5. Clustering K-Means",
    "📉 6. PCA (Redução Dimensional)"
    ])

    # =====================================
    # VARIÁVEIS GLOBAIS (CORREÇÃO DO ERRO)
    # =====================================
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
>>>>>>> 88a5fd1 (feat: ferramenta estatística COMPLETA v3.0)


    
    # =====================================
    # ANÁLISE 1: DESCRITIVA (ESTATÍSTICAS BÁSICAS)
    # =====================================
    if analise_tipo == "📊 1. Análise Descritiva":
        st.header("📊 1. Análise Descritiva Completa")
        st.markdown("**CONCEITO:** Resumo estatístico + normalidade + visualizações diagnósticas")
        
        # Seleciona variáveis numéricas
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        col1, col2 = st.columns(2)
        
        with col1:
            vars_analise = st.multiselect(
                "🔢 Variáveis para análise", 
                num_cols, 
                default=num_cols[:3]  # Pega primeiras 3 automaticamente
            )
        with col2:
            var_alvo = st.selectbox("📍 Variável alvo (gráficos)", num_cols)
        
        if st.button("🔬 Executar Análise Descritiva", type="primary") and vars_analise:
            st.subheader("📈 Tabela de Estatísticas Descritivas")
            
            # Tabela completa (transposta para melhor visualização)
            desc_stats = df[vars_analise].describe().round(3).T
            st.dataframe(desc_stats, use_container_width=True)
            
            # 💡 INTERPRETAÇÃO AUTOMÁTICA
            st.subheader("💡 Interpretação Estatística Automática")
            for var in vars_analise:
                # Cálculos chave
                media = df[var].mean()
                dp = df[var].std()
                cv = (dp/media)*100 if media != 0 else 0  # Coeficiente de variação
                
                # Teste de normalidade (Shapiro-Wilk)
                stat, p_shapiro = stats.shapiro(df[var].dropna()[:5000])  # Limita amostra
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Média", f"{media:.2f}")
                col2.metric("DP", f"{dp:.2f}")
                col3.metric("CV%", f"{cv:.1f}%")
                
                # Interpretação normalidade
                if p_shapiro < 0.05:
                    st.error(f"❌ **{var}:** Não normal (Shapiro p={p_shapiro:.4f})")
                    st.info("→ Use testes não-paramétricos (Mann-Whitney, Kruskal-Wallis)")
                else:
                    st.success(f"✅ **{var}:** Normal (Shapiro p={p_shapiro:.4f})")
                    st.info("→ Use testes paramétricos (t-test, ANOVA)")
            
            # 📊 VISUALIZAÇÕES DIAGNÓSTICAS
            st.subheader("📊 Visualizações Diagnósticas")
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('📈 Histograma', '📦 Boxplot', '📊 QQ-Plot Normalidade', '🎲 Densidade'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Histograma
            fig.add_trace(go.Histogram(x=df[var_alvo], name="Histograma", nbinsx=30), row=1, col=1)
            
            # Boxplot
            fig.add_trace(go.Box(y=df[var_alvo], name="Boxplot"), row=1, col=2)
            
            # QQ-Plot (diagnóstico normalidade)
            try:
                from scipy.stats import norm, probplot
                qq_data = df[var_alvo].dropna()
                (osm, osr), (slope, intercept, r) = probplot(qq_data, dist="norm", plot=False)
                fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers+lines', 
                           name="QQ-Plot", line=dict(color='blue')), row=2, col=1)
            except:
                fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', 
                           name="QQ-Plot (simplificado)"), row=2, col=1)
            # KDE (densidade)
            fig.add_trace(go.Histogram(x=df[var_alvo], histnorm='probability density'), row=2, col=2)
            
            fig.update_layout(height=600, showlegend=False, title_text="Diagnósticos Visuais")
            st.plotly_chart(fig, use_container_width=True)
    
    # =====================================
    # ANÁLISE 2: CORRELAÇÃO
    # =====================================
    elif analise_tipo == "🔗 2. Correlação":
        st.header("🔗 2. Matriz de Correlação")
        st.markdown("""
        **CONCEITO:** Mede relação linear entre variáveis (Pearson r ∈ [-1,1])
        - **r > 0.7 ou r < -0.7:** Correlação forte
        - **p-valor < 0.05:** Significativa estatisticamente
        """)
        
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(num_cols) >= 2 and st.button("🔗 Calcular Correlação", type="primary"):
            # Matriz Pearson
            corr_matrix = df[num_cols].corr()
            
            # Heatmap interativo
            fig = px.imshow(
                corr_matrix.round(3),
                title="🔥 Matriz de Correlação Pearson",
                color_continuous_scale='RdBu_r',  # Vermelho=negativo, Azul=positivo
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlações fortes
            st.subheader("🚨 Correlações Fortes |r| > 0.7")
            mask = (abs(corr_matrix) > 0.7) & (corr_matrix != 1.0)
            strong_corr = corr_matrix[mask].stack().reset_index()
            strong_corr.columns = ['Variável 1', 'Variável 2', 'r_Pearson']
            strong_corr['Força'] = pd.cut(abs(strong_corr['r_Pearson']), 
                                        bins=[0.7, 0.8, 0.9, 1], 
                                        labels=['Forte', 'Muito Forte', 'Perfeita'])
            st.dataframe(strong_corr.round(3))

    # =====================================
    # ANÁLISE 3: DENGUE/SAÚDE PÚBLICA
    # =====================================

    elif analise_tipo == "🏥 3. Análise Saúde":
        st.header("🏥 3. Análise Saúde - TOP Regiões")
        
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        
        col1, col2 = st.columns(2)
        with col1:
            col_casos = st.selectbox("📊 Casos/Óbitos", num_cols)
        with col2:
            col_regiao = st.selectbox("🏛️ Município/UF", cat_cols)
        
        if st.button("🚨 TOP 10 Regiões", type="primary"):
            # TOP 10 genérico
            top10 = df.nlargest(10, col_casos)[[col_regiao, col_casos]].round(1)
            top10.columns = ['Região', 'Valor']
            
            st.subheader("🔥 TOP 10 Regiões MAIS AFETADAS")
            st.dataframe(top10, use_container_width=True)
            
            # Gráfico
            fig = px.bar(top10, x='Região', y='Valor', 
                        title=f"TOP 10 - {col_casos}", 
                        color='Valor')
            st.plotly_chart(fig, use_container_width=True)
    
    # =====================================
    # ANÁLISE 4: TESTES ESTATÍSTICOS (CORRIGIDO)
    # =====================================
    elif analise_tipo == "📈 4. Testes Estatísticos":
        st.header("📈 4. Testes Inferenciais")
        st.markdown("**CONCEITO:** Verifica hipóteses (H0 vs H1) com p-valor < 0.05")
        
        col1, col2 = st.columns(2)
        with col1:
            teste_tipo = st.selectbox("Teste", ["t-test (2 grupos)", "ANOVA (3+ grupos)", "Qui-Quadrado"])
        with col2:
            var_resposta = st.selectbox("📊 Variável numérica", num_cols)
        
        if teste_tipo == "t-test (2 grupos)" and st.button("🔬 Executar t-test", type="primary"):
            grupo_var = st.selectbox("🏷️ Variável grupos", cat_cols)
            grupos = df[grupo_var].dropna().unique()[:2]  # Primeiros 2 grupos
            
            if len(grupos) >= 2:
                grupo1 = df[df[grupo_var] == grupos[0]][var_resposta].dropna()
                grupo2 = df[df[grupo_var] == grupos[1]][var_resposta].dropna()
                
                if len(grupo1) > 1 and len(grupo2) > 1:
                    t_stat, p_val = stats.ttest_ind(grupo1, grupo2)
                    
                    col1, col2 = st.columns(2)
                    col1.metric("📊 t-statistic", f"{t_stat:.3f}")
                    col2.metric("🎯 p-valor", f"{p_val:.4f}")
                    
                    st.subheader("💡 Interpretação")
                    if p_val < 0.05:
                        st.error(f"🚨 **REJEITA H0** (p={p_val:.4f})")
                        st.success(f"✅ {grupos[0]} **≠** {grupos[1]} em {var_resposta}")
                    else:
                        st.info(f"ℹ️ **NÃO rejeita H0** (p={p_val:.4f})")
                        st.warning(f"{grupos[0]} **≈** {grupos[1]} em {var_resposta}")
                else:
                    st.warning("❌ Poucos dados em um dos grupos")
            else:
                st.warning("❓ Selecione variável com ≥2 grupos")

    elif analise_tipo == "🎭 5. Clustering K-Means":
        st.header("🎭 5. Clustering Automático")
        st.markdown("**CONCEITO:** Agrupa observações similares automaticamente")
        
        vars_cluster = st.multiselect("🔢 Variáveis para cluster", num_cols, default=num_cols[:3])
        n_clusters = st.slider("Número de clusters", 2, 8, 4)
        
        if st.button("🤖 Executar Clustering", type="primary") and len(vars_cluster)>=2:
            # Padroniza + clusteriza
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df[vars_cluster].dropna())
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            df_cluster = df.dropna(subset=vars_cluster).copy()
            df_cluster['Cluster'] = clusters
            
            st.subheader("📊 Resultado Clustering")
            st.dataframe(df_cluster.groupby('Cluster')[vars_cluster].mean().round(2))
            
            # Gráfico 2D
            fig = px.scatter(df_cluster, x=vars_cluster[0], y=vars_cluster[1], 
                            color='Cluster', title="Clusters Automáticos")
            st.plotly_chart(fig)
            
    elif analise_tipo == "📉 6. PCA (Redução Dimensional)":
        st.header("📉 6. PCA - Redução Dimensional")
        vars_pca = st.multiselect("🔢 Variáveis", num_cols, default=num_cols[:4])
        
        if st.button("📉 Executar PCA", type="primary") and len(vars_pca)>=2:
            from sklearn.decomposition import PCA
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df[vars_pca].dropna())
            
            pca = PCA()
            X_pca = pca.fit_transform(X_scaled)
            
            st.subheader("📊 Variância Explicada")
            var_exp = pd.DataFrame({
                'Componente': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                'Variância %': (pca.explained_variance_ratio_*100).round(1)
            })
            st.dataframe(var_exp)
            
            fig = px.scatter(x=X_pca[:,0], y=X_pca[:,1], 
                            title="PCA 2D - Primeiros 2 Componentes")
            st.plotly_chart(fig)


<<<<<<< HEAD
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

=======
   
    # =====================================
    # BARRA LATERAL: EXPORT
    # =====================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 Exportar")
    csv_bytes = convert_df(df)
    st.sidebar.download_button(
        "📥 Dados Originais CSV",
        csv_bytes,
        "dados_originais.csv",
        "text/csv"
    )

# =====================================
# ESTADO INICIAL (SEM DADOS)
# =====================================
>>>>>>> 88a5fd1 (feat: ferramenta estatística COMPLETA v3.0)
else:
    st.info("""
    👆 **Carregue um dataset CSV/Excel para começar!**
    
    **Exemplos recomendados para estudo:**
    - Iris (classificação)
    - Boston Housing (regressão)  
    - Titanic (análise exploratória)
    - Qualquer base com ≥3 colunas numéricas
    """)
