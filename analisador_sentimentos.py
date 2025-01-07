# Importa as bibliotecas necessárias.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Define uma função para garantir acessibilidade dos gráficos com Matplotlib.
def configurar_acessibilidade_matplotlib():
    # Ativa o modo interativo para garantir que as configurações sejam aplicadas.
    plt.ion()
    # Define o tamanho da fonte para os rótulos dos eixos.
    plt.rcParams['axes.labelsize'] = 14
    # Define o tamanho da fonte para os números dos ticks dos eixos.
    plt.rcParams['xtick.labelsize'] = 12
    # Define o tamanho da fonte para os números dos ticks dos eixos.
    plt.rcParams['ytick.labelsize'] = 12
    # Define o tamanho da fonte para o título do gráfico.
    plt.rcParams['axes.titlesize'] = 16


# Carrega o conjunto de dados de exemplo.  Substitua por seu conjunto de dados real.
# Certifique-se de que os dados estejam em formato CSV com colunas "texto" e "sentimento".
try:
    dados = pd.read_csv("dados_sentimentos.csv")  # Substitua "dados_sentimentos.csv" pelo nome do seu arquivo
except FileNotFoundError:
    print("Arquivo de dados não encontrado. Certifique-se de que 'dados_sentimentos.csv' esteja no mesmo diretório.")
    exit() # Encerra a execução do script se o arquivo não for encontrado
# Define as colunas de texto e sentimento.
texto = dados["texto"]
sentimento = dados["sentimento"]


# Divide os dados em conjuntos de treinamento e teste.
texto_treinamento, texto_teste, sentimento_treinamento, sentimento_teste = train_test_split(
    texto, sentimento, test_size=0.2, random_state=42
)

# Cria um vetorizador TF-IDF para converter o texto em recursos numéricos.
vetorizador = TfidfVectorizer()
# Ajusta o vetorizador aos dados de treinamento e transforma-os.
texto_treinamento_vec = vetorizador.fit_transform(texto_treinamento)
# Transforma os dados de teste usando o vetorizador ajustado.
texto_teste_vec = vetorizador.transform(texto_teste)


# Cria um modelo de regressão logística para classificação de sentimentos.
modelo = LogisticRegression()
# Treina o modelo nos dados de treinamento vetorizados.
modelo.fit(texto_treinamento_vec, sentimento_treinamento)

# Faz previsões nos dados de teste vetorizados.
previsoes = modelo.predict(texto_teste_vec)


# Avalia o desempenho do modelo.
acuracia = accuracy_score(sentimento_teste, previsoes)
relatorio_classificacao = classification_report(sentimento_teste, previsoes)


# Imprime as métricas de avaliação.
print(f"Acurácia: {acuracia}")
print(f"Relatório de Classificação:\n{relatorio_classificacao}")


# Configura a acessibilidade dos gráficos Matplotlib.
configurar_acessibilidade_matplotlib()

# Cria um gráfico de barras para visualizar a distribuição das classes.
plt.figure(figsize=[8, 6])
sns.countplot(x=dados['sentimento'])
plt.title("Distribuição de Sentimentos")
plt.xlabel("Sentimento")
plt.ylabel("Contagem")
plt.savefig('grafico_sentimentos.png')
plt.show()

# Exemplo de uso do modelo para prever o sentimento de um novo texto.
novo_texto = ["Este filme é ótimo!"]
novo_texto_vec = vetorizador.transform(novo_texto)
previsao_novo_texto = modelo.predict(novo_texto_vec)
print(f"Sentimento previsto para '{novo_texto[0]}': {previsao_novo_texto[0]}")