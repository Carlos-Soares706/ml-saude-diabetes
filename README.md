Atividade Final: Machine Learning Aplicado à Saúde - Diabetes

Este projeto foi desenvolvido como parte da Atividade Final da disciplina de Machine Learning Aplicado à Saúde. O objetivo é demonstrar a aplicação de técnicas de aprendizado supervisionado e não supervisionado em um dataset real da área de saúde.

🧠 Tema Escolhido

Dataset: Diabetes (disponível no sklearn.datasets.load_diabetes)

O dataset contém dados de 442 pacientes de diabetes, com 10 variáveis preditoras (idade, sexo, BMI, pressão sanguínea, etc.) e uma variável alvo que é uma medida quantitativa da progressão da doença um ano após a linha de base.

🛠️ Tecnologias Utilizadas

•
Python

•
Streamlit: Para criar a interface web interativa.

•
Scikit-learn: Para os modelos de Machine Learning (Regressão Linear e K-Means).

•
Pandas e NumPy: Para manipulação de dados.

•
Matplotlib e Seaborn: Para visualização e EDA.

🚀 Como Rodar o Projeto Localmente

1.
Clone o repositório: ```bash git clone [LINK DO SEU REPOSITÓRIO NO GITHUB] cd ml-saude-diabetes ```

2.
Crie e ative o ambiente virtual: ```bash python3 -m venv venv source venv/bin/activate ```

3.
Instale as dependências: ```bash pip install -r requirements.txt ```

4.
Execute o Streamlit: ```bash streamlit run app.py ```

O aplicativo será aberto automaticamente no seu navegador.

📊 Estrutura da Aplicação

A aplicação Streamlit (app.py) está dividida em três seções principais, acessíveis pelo menu lateral:

1.
Exploração dos Dados (EDA): Visualização inicial dos dados, estatísticas e distribuição da variável alvo.

2.
Aprendizagem Supervisionada: Aplicação de Regressão Linear para prever a progressão da doença e análise das métricas (MSE e R²).

3.
Aprendizagem Não Supervisionada: Aplicação do algoritmo K-Means para segmentar os pacientes em grupos (clusters) e análise do perfil de cada grupo.

📝 Observações

O código foi escrito de forma clara e com comentários para facilitar a compreensão de cada etapa, mantendo a simplicidade e a funcionalidade exigida pela atividade. A escolha dos modelos (Regressão Linear e K-Means) foi baseada na simplicidade e na facilidade de interpretação dos resultados.

