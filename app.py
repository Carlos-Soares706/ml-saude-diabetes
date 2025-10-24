
# Importando as bibliotecas que vamos usar
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# --- Configurações Iniciais ---
st.set_page_config(
    page_title="ML Aplicado à Saúde - Diabetes",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título da Aplicação
st.title("Atividade Final: Machine Learning Aplicado à Saúde")
st.subheader("Análise e Modelagem de Dados de Diabetes (Dataset Sklearn)")

# --- 1. Carregamento e Preparação dos Dados ---

@st.cache_data
def load_data():
    """Carrega o dataset de diabetes do sklearn e transforma em DataFrame."""
    # Carregando o dataset. Simples assim.
    diabetes = load_diabetes(as_frame=True)
    df = diabetes.frame
    
    # A variável 'target' é a medida quantitativa da progressão da doença.
    # Vamos renomear para Target_Progressao para ficar mais claro.
    df.rename(columns={'target': 'Target_Progressao'}, inplace=True)
    
    return df

df_diabetes = load_data()

# --- Conteúdo da Aplicação Streamlit ---

# Sidebar para navegação (como um estudante faria para organizar o projeto)
st.sidebar.title("Menu de Navegação")
menu_selection = st.sidebar.radio(
    "Escolha a seção:",
    ("1. Exploração dos Dados (EDA)", "2. Aprendizagem Supervisionada", "3. Aprendizagem Não Supervisionada")
)

# --- 1. Exploração dos Dados (EDA) ---
if menu_selection == "1. Exploração dos Dados (EDA)":
    st.header("1. Exploração dos Dados (EDA)")
    st.markdown("""
        Nesta seção, vamos dar uma olhada inicial nos dados para entender o que temos.
        O dataset de diabetes possui 10 variáveis preditoras (features) e uma variável alvo (target).
        As features são padronizadas e centradas, então não precisamos nos preocupar com a escala.
    """)
    
    st.subheader("Amostra dos Dados")
    st.dataframe(df_diabetes.head())
    
    st.subheader("Estatísticas Descritivas")
    st.dataframe(df_diabetes.describe().T)
    
    st.subheader("Visualização da Variável Alvo")
    # Um histograma simples para ver a distribuição da progressão da doença
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df_diabetes['Target_Progressao'], kde=True, ax=ax)
    ax.set_title("Distribuição da Progressão da Doença (Target)")
    ax.set_xlabel("Target_Progressao")
    st.pyplot(fig)
    
    st.subheader("Relação entre Features e Target")
    # Vamos escolher uma feature para plotar a relação com o target
    feature_to_plot = st.selectbox(
        "Selecione uma feature para ver a correlação com o Target:",
        df_diabetes.columns[:-1] # Exclui o 'Target_Progressao'
    )
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x=df_diabetes[feature_to_plot], y=df_diabetes['Target_Progressao'], ax=ax)
    ax.set_title(f"Target vs. {feature_to_plot}")
    st.pyplot(fig)

# --- 2. Aprendizagem Supervisionada (Regressão Linear) ---
elif menu_selection == "2. Aprendizagem Supervisionada":
    st.header("2. Aprendizagem Supervisionada: Regressão Linear")
    st.markdown("""
        Como a nossa variável alvo ('Target_Progressao') é **quantitativa** e **contínua**,
        o problema se enquadra em um modelo de **Regressão**.
        Vamos usar o modelo mais básico e interpretável: **Regressão Linear Múltipla**.
        O objetivo é prever a progressão da doença com base nas 10 features.
    """)
    
    # 1. Preparação
    X = df_diabetes.drop('Target_Progressao', axis=1)
    y = df_diabetes['Target_Progressao']
    
    # Dividindo os dados em treino e teste (padrão 80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.subheader("Divisão dos Dados")
    st.info(f"Dados de Treino: {len(X_train)} amostras | Dados de Teste: {len(X_test)} amostras")
    
    # 2. Treinamento do Modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 3. Previsão
    y_pred = model.predict(X_test)
    
    # 4. Avaliação
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.subheader("Métricas de Avaliação")
    st.metric("Erro Quadrático Médio (MSE)", f"{mse:.2f}")
    st.metric("Coeficiente de Determinação (R²)", f"{r2:.2f}")
    
    st.markdown("""
        **Interpretação:**
        - **MSE:** Quanto menor, melhor. É a média do quadrado dos erros de previsão.
        - **R²:** Indica a proporção da variância na variável dependente que é previsível a partir das variáveis independentes. Um R² de 0.53 significa que 53% da variação na progressão da doença é explicada pelo nosso modelo. Não é perfeito, mas é um bom começo!
    """)
    
    st.subheader("Coeficientes do Modelo")
    # Mostrando quais features o modelo achou mais importantes
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coeficiente': model.coef_
    }).sort_values(by='Coeficiente', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Coeficiente', y='Feature', data=coef_df, ax=ax)
    ax.set_title("Importância das Features (Coeficientes da Regressão)")
    st.pyplot(fig)
    
    st.markdown("""
        Os coeficientes mostram a influência de cada feature na progressão da doença.
        Features com coeficientes positivos e altos aumentam a progressão, e vice-versa.
    """)

# --- 3. Aprendizagem Não Supervisionada (K-Means) ---
elif menu_selection == "3. Aprendizagem Não Supervisionada":
    st.header("3. Aprendizagem Não Supervisionada: K-Means")
    st.markdown("""
        Aqui, o objetivo é encontrar **grupos (clusters)** naturais nos dados, sem usar a variável alvo.
        Vamos usar o algoritmo **K-Means** para segmentar os pacientes com base em suas 10 features.
        Isso pode nos ajudar a identificar perfis de pacientes.
    """)
    
    # 1. Preparação
    # K-Means é sensível à escala, mas as features já estão padronizadas no dataset do sklearn.
    # No entanto, vamos re-escalar por boa prática, excluindo o target.
    features_for_clustering = df_diabetes.drop('Target_Progressao', axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_for_clustering)
    
    # 2. Escolha do K (Número de Clusters)
    # Vamos deixar o usuário escolher, mas sugerir um padrão.
    n_clusters = st.slider("Escolha o número de clusters (K):", min_value=2, max_value=8, value=3)
    
    # 3. Treinamento
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # n_init=10 para evitar warning
    df_diabetes['Cluster'] = kmeans.fit_predict(X_scaled)
    
    st.subheader(f"Resultado do Clustering K-Means (K={n_clusters})")
    st.info(f"O K-Means identificou {n_clusters} perfis de pacientes.")
    
    # 4. Análise dos Clusters
    # Vamos ver o tamanho de cada cluster
    cluster_counts = df_diabetes['Cluster'].value_counts().sort_index()
    st.dataframe(cluster_counts.rename("Tamanho do Cluster"))
    
    st.subheader("Características Médias de Cada Cluster")
    # Agrupando os dados pelo cluster e vendo a média de cada feature
    cluster_profile = df_diabetes.groupby('Cluster').mean().T
    st.dataframe(cluster_profile)
    
    st.markdown("""
        **Interpretação:**
        - Analisando a tabela acima, podemos ver as características médias de cada grupo.
        - Por exemplo, o **Cluster 2** (se for o caso) pode ter a maior média em 'Target_Progressao', indicando que esse grupo de pacientes tem a progressão da doença mais avançada.
        - O objetivo é dar nomes a esses grupos (ex: 'Risco Baixo', 'Risco Moderado', 'Risco Alto') com base nas médias das features.
    """)
    
    st.subheader("Visualização dos Clusters (Target vs. BMI)")
    # Para visualizar, vamos usar duas features (BMI e Target)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x='bmi', 
        y='Target_Progressao', 
        hue='Cluster', 
        data=df_diabetes, 
        palette='viridis', 
        style='Cluster',
        s=100, # Tamanho do ponto
        ax=ax
    )
    ax.set_title(f"Clusters de Pacientes (K={n_clusters})")
    ax.set_xlabel("BMI (Índice de Massa Corporal)")
    ax.set_ylabel("Progressão da Doença (Target)")
    st.pyplot(fig)

# --- Fim do Código ---
st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido por [Seu Nome Aqui] para a Atividade Final de ML.")

