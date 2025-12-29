"""
🧮 FERRAMENTA DE ANÁLISE ESTATÍSTICA AVANÇADA
Autor: Arthur (Estudos Ciência de Dados)
Objetivo: Análise descritiva + testes inferenciais + ML com interpretação automática
Skills: Estatística aplicada, Streamlit, Pandas, SciPy, Scikit-learn
"""


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

@st.cache_data
def compute_descritiva(df, vars_analise):
    """Cache estatísticas descritivas"""
    return df[vars_analise].describe().round(3)

@st.cache_data
def compute_correlacao(df, num_cols):
    """Cache matriz correlação"""
    return df[num_cols].corr()

@st.cache_data
def top_regioes(df, col_casos, col_regiao):
    return df.nlargest(10, col_casos)[[col_regiao, col_casos]].round(1)
    
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


# Permite usar dataset gerado via botão de teste (Iris)
if uploaded_file is not None or 'uploaded_df' in st.session_state:
    # Fonte: upload de arquivo ou dataset gerado em sessão
    if 'uploaded_df' in st.session_state:
        df = st.session_state['uploaded_df']
    else:
        # Detecta formato automaticamente
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        else:
            df = pd.read_excel(uploaded_file)

    # =====================================
    # VARIÁVEIS GLOBAIS
    # =====================================
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # ✅ CONFIRMAÇÃO VISUAL
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Linhas", f"{df.shape[0]:,}")
    col2.metric("Colunas", df.shape[1])
    col3.metric("Numéricas", len(df.select_dtypes(include=[np.number]).columns))
    col4.metric("Categóricas", len(df.select_dtypes(exclude=[np.number]).columns))
    
    st.dataframe(df.head(10), width="stretch")


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
            desc_stats = compute_descritiva(df, vars_analise)
            st.dataframe(desc_stats, width="stretch")
            
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
            st.plotly_chart(fig, width="stretch")
    
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
            corr_matrix = compute_correlacao(df, num_cols)
            
            # Heatmap interativo
            fig = px.imshow(
                corr_matrix.round(3),
                title="🔥 Matriz de Correlação Pearson",
                color_continuous_scale='RdBu_r',  # Vermelho=negativo, Azul=positivo
                aspect="auto"
            )
            st.plotly_chart(fig, width="stretch")
            
            # Correlações fortes
            st.subheader("🚨 Correlações Fortes |r| > 0.7")
            mask = (abs(corr_matrix) > 0.7) & (corr_matrix != 1.0)
            strong_corr = corr_matrix[mask].stack().reset_index()
            strong_corr.columns = ['Variável 1', 'Variável 2', 'r_Pearson']
            strong_corr['Força'] = pd.cut(abs(strong_corr['r_Pearson']), 
                                        bins=[0.7, 0.8, 0.9, 1], 
                                        labels=['Forte', 'Muito Forte', 'Perfeita'])
            st.dataframe(strong_corr.round(3))


    # ANÁLISE 3: DENGUE/SAÚDE PÚBLICA
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
            top10 = top_regioes(df, col_casos, col_regiao)
            top10.columns = ['Região', 'Valor']
            
            st.subheader("🔥 TOP 10 Regiões MAIS AFETADAS")
            st.dataframe(top10, width="stretch")
            
            # Gráfico
            fig = px.bar(top10, x='Região', y='Valor', 
                        title=f"TOP 10 - {col_casos}", 
                        color='Valor')
            st.plotly_chart(fig, width="stretch")

    # =====================================
    # ANÁLISE 4: TESTES ESTATÍSTICOS
    # =====================================
    elif analise_tipo == "📈 4. Testes Estatísticos":
        st.header("📈 4. Testes Inferenciais")
        st.markdown("**CONCEITO:** Verifica hipóteses (H0 vs H1) com p-valor < 0.05")
        
        col1, col2 = st.columns(2)
        with col1:
            teste_tipo = st.selectbox("Teste", ["t-test (2 grupos)", "ANOVA (3+ grupos)", "Qui-Quadrado"])
        with col2:
            var_resposta = st.selectbox("📊 Variável numérica", num_cols)
        
        if teste_tipo == "t-test (2 grupos)":
            grupo_var = st.selectbox("🏷️ Variável grupos", cat_cols)
            if st.button("🔬 Executar t-test", type="primary"):
                grupos = df[grupo_var].dropna().unique()[:2]
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
        
        elif teste_tipo == "ANOVA (3+ grupos)":
            grupo_var = st.selectbox("🏷️ Variável grupos", cat_cols)
            grupos = df[grupo_var].dropna().unique()
            
            if len(grupos) >= 3 and st.button("🔬 Executar ANOVA", type="primary"):
                grupo_dados = [df[df[grupo_var]==g][var_resposta].dropna() for g in grupos]
                f_stat, p_val = stats.f_oneway(*grupo_dados)
                
                col1, col2 = st.columns(2)
                col1.metric("F-statistic", f"{f_stat:.3f}")
                col2.metric("p-valor", f"{p_val:.4f}")
                
                if p_val < 0.05:
                    st.error(f"🚨 **REJEITA H0** - Pelo menos 1 grupo difere!")
                else:
                    st.success("ℹ️ **NÃO rejeita H0** - Grupos similares")
        
        elif teste_tipo == "Qui-Quadrado":
            col1_var = st.selectbox("🏷️ Variável 1 (categórica)", cat_cols)
            col2_var = st.selectbox("🏷️ Variável 2 (categórica)", cat_cols)
            
            if st.button("🔬 Executar Qui-Quadrado", type="primary"):
                contingency = pd.crosstab(df[col1_var], df[col2_var])
                chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
                
                col1, col2 = st.columns(2)
                col1.metric("χ²", f"{chi2:.3f}")
                col2.metric("p-valor", f"{p_val:.4f}")
                
                if p_val < 0.05:
                    st.error("🚨 **REJEITA H0** - Variáveis são dependentes!")
                else:
                    st.success("ℹ️ **NÃO rejeita** - Variáveis independentes")

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

else:
    st.info("👆 **Carregue CSV/Excel OU teste com dados automáticos**")
    
    if st.button("🧪 Gerar Iris Dataset (teste)", type="primary"):
        st.info("🔄 Carregando Iris Dataset...")
        from sklearn.datasets import load_iris
        iris = load_iris()
        df_test = pd.DataFrame(iris.data, columns=iris.feature_names)
        df_test['target'] = iris.target
        st.session_state['uploaded_df'] = df_test
        st.success("✅ **Iris Dataset carregado!** (150 amostras, 5 colunas)")
        st.rerun()
