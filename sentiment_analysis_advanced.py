# ==============================================================================
# Versão: 0.2
# Autor: Eliezer Tavares de Oliveira
# Site: www.eliezertavaresdeoliveira.com
# Descrição:
# Este script implementa um pipeline completo para análise de sentimento,
# utilizando Regressão Logística e TF-IDF. Inclui geração automática de um CSV
# com 27 classes de sentimentos (1.000 entradas por classe, total 27.000 entradas)
# caso o arquivo de dados não exista, carregamento de dados, pré-processamento,
# treinamento de modelo com otimização de hiperparâmetros, avaliação de desempenho,
# persistência de modelo e um módulo de previsão interativo. Suporta relatórios
# em TXT, HTML e PDF, com ênfase em acessibilidade.
#
# This script implements a complete pipeline for sentiment analysis using
# Logistic Regression and TF-IDF. It includes automatic generation of a CSV
# with 27 sentiment classes (1,000 entries per class, total 27,000 entries)
# if the data file doesn't exist, data loading, preprocessing, model training
# with hyperparameter optimization, performance evaluation, model persistence,
# and an interactive prediction module. Supports reports in TXT, HTML, and PDF,
# with a focus on accessibility.
# ==============================================================================

# 1. Importa as bibliotecas necessárias / Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import argparse
import joblib
import os
import sys
import random
from typing import Optional, Tuple
from jinja2 import Environment, FileSystemLoader
import datetime

# Tenta importar pypandoc, mas não falha se não estiver instalado / Try importing pypandoc, but don't fail if not installed
try:
    import pypandoc
except ImportError:
    pypandoc = None

# 2. Definição de Constantes e Metadados do Projeto / Project Metadata and Constants Definition
PROJECT_METADATA = {
    "version": "0.2",
    "author": "Eliezer Tavares de Oliveira",
    "website": "www.eliezertavaresdeoliveira.com",
    "project_name": "Análise de Sentimento com ML"
}

# --- Configurações de Arquivo e Colunas / File and Column Configuration ---
DATA_FILE_DEFAULT = 'dados_sentimentos.csv'
TEXT_COLUMN = 'texto'
SENTIMENT_COLUMN = 'sentimento'
MODEL_SAVE_PATH = 'modelo_sentimentos.pkl'
VECTORIZER_SAVE_PATH = 'vetorizador.pkl'
GRAPH_IMAGE_PATH = 'grafico_sentimentos.png'
GRAPH_DESC_PATH = 'grafico_sentimentos_descricao.txt'
REPORT_TEXT_PATH = 'relatorio_classificacao.txt'
REPORT_HTML_PATH = 'relatorio_classificacao.html'
REPORT_PDF_PATH = 'relatorio_classificacao.pdf'
FEATURE_IMPORTANCE_PATH = 'importancia_features.csv'
LOG_FILE_PATH = 'sentiment_analysis.log'

# --- Configurações do Modelo / Model Configuration ---
TEST_SIZE_RATIO = 0.2
RANDOM_SEED = 42
MAX_FEATURES_TFIDF = 5000
MAX_ITER_LR = 1000  # Aumentado para melhor convergência / Increased for better convergence

# --- Configurações para Geração de Dados Sintéticos / Synthetic Data Generation Configuration ---
ENTRADAS_POR_CLASSE = 1000
CLASSES = [
    'admiração', 'alegria', 'alívio', 'amor', 'antecipação', 'apreensão', 'aprovação',
    'confusão', 'curiosidade', 'decepção', 'desaprovação', 'desgosto', 'diversão',
    'entusiasmo', 'gratidão', 'inveja', 'medo', 'nojo', 'nostalgia', 'orgulho', 'raiva',
    'remorso', 'surpresa', 'tristeza', 'vergonha', 'neutro', 'otimismo'
]
ITENS = [
    "produto", "atendimento", "serviço", "entrega", "item", "compra", "pedido",
    "experiência", "loja", "suporte"
]
PADROES = {
    'admiração': ["Que {item} impressionante, fiquei admirado!", "O {item} é de tirar o fôlego!"],
    'alegria': ["Adorei o {item}, estou muito feliz!", "Que {item} incrível, pura alegria!"],
    'alívio': ["Ufa, o {item} chegou a tempo, que alívio!", "Fiquei aliviado com o {item}."],
    'amor': ["Amei o {item}, é perfeito!", "Tenho um carinho especial por esse {item}."],
    'antecipação': ["Mal posso esperar para usar o {item}!", "Ansioso pelo próximo {item}!"],
    'apreensão': ["Estou preocupado com o {item}, será que vai dar certo?", "O {item} me deixa apreensivo."],
    'aprovação': ["O {item} está ótimo, aprovo totalmente!", "Muito bom, o {item} merece aprovação!"],
    'confusão': ["Não entendi o {item}, está confuso.", "O {item} me deixou meio perdido."],
    'curiosidade': ["Fiquei curioso sobre o {item}, parece interessante!", "Quero saber mais sobre esse {item}!"],
    'decepção': ["Que decepção, o {item} não era o que eu esperava.", "Fiquei desapontado com o {item}."],
    'desaprovação': ["Não gostei do {item}, desaprovo!", "O {item} não está à altura, desaprovado."],
    'desgosto': ["Que nojo, o {item} foi uma péssima experiência!", "Senti desgosto com o {item}."],
    'diversão': ["O {item} foi super divertido, adorei!", "Que {item} engraçado, me diverti muito!"],
    'entusiasmo': ["Estou entusiasmado com o {item}, é incrível!", "O {item} me deixou super animado!"],
    'gratidão': ["Sou grato pelo {item}, muito obrigado!", "O {item} foi um presente, gratidão!"],
    'inveja': ["Queria ter esse {item}, que inveja!", "Fiquei com inveja desse {item}."],
    'medo': ["O {item} me deixou com medo, não sei se confio.", "Tenho receio do {item}."],
    'nojo': ["Que nojo, o {item} foi horrível!", "Senti repulsa com o {item}."],
    'nostalgia': ["O {item} me lembrou os velhos tempos, que nostalgia!", "Senti saudades com esse {item}."],
    'orgulho': ["Estou orgulhoso do {item}, ficou perfeito!", "O {item} é motivo de orgulho!"],
    'raiva': ["O {item} foi horrível, que raiva!", "Estou indignado com o {item}!"],
    'remorso': ["Me arrependi de comprar o {item}.", "Sinto remorso por causa do {item}."],
    'surpresa': ["Fiquei surpreso com o {item}, uau!", "Não esperava que o {item} fosse tão bom!"],
    'tristeza': ["Que tristeza, o {item} chegou quebrado.", "Fiquei triste com o {item}."],
    'vergonha': ["Que vergonha, o {item} foi um fiasco!", "Senti vergonha alheia com o {item}."],
    'neutro': ["O {item} chegou no prazo, tudo certo.", "{Item} conforme a descrição."],
    'otimismo': ["Acho que o {item} será ótimo, estou otimista!", "O {item} promete, boas expectativas!"]
}

# 3. Configura o logging para console e arquivo / Configure logging for console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding='utf-8'),  # Salva logs em arquivo / Saves logs to file
        logging.StreamHandler(sys.stdout)  # Exibe logs no console / Displays logs in console
    ]
)
logger = logging.getLogger(__name__)

def gerar_dados_sinteticos(arquivo: str, entradas_por_classe: int = ENTRADAS_POR_CLASSE) -> None:
    """
    Gera um arquivo CSV sintético com 27 classes de sentimentos e entradas_por_classe por classe.
    
    Generates a synthetic CSV file with 27 sentiment classes and entradas_por_classe per class.
    
    Args:
        arquivo (str): Caminho do arquivo CSV a ser gerado / Path to the CSV file to be generated
        entradas_por_classe (int): Número de entradas por classe / Number of entries per class
    """
    logger.info(f"Gerando dados sintéticos para {arquivo} com {entradas_por_classe} entradas por classe... / "
                f"Generating synthetic data for {arquivo} with {entradas_por_classe} entries per class...")
    
    dados = []
    for classe in CLASSES:
        padroes_classe = PADROES[classe]
        for _ in range(entradas_por_classe):
            padrao = random.choice(padroes_classe)
            item = random.choice(ITENS)
            texto = padrao.format(item=item, Item=item.capitalize())
            dados.append((texto, classe))
        logger.info(f"Geradas {entradas_por_classe} entradas para a classe '{classe}' / "
                    f"Generated {entradas_por_classe} entries for class '{classe}'")
    
    # Embaralha as entradas / Shuffle entries
    random.shuffle(dados)
    
    # Cria e salva o DataFrame / Create and save DataFrame
    df = pd.DataFrame(dados, columns=[TEXT_COLUMN, SENTIMENT_COLUMN])
    df.to_csv(arquivo, index=False, encoding='utf-8')
    logger.info(f"Arquivo CSV salvo em {arquivo} com {len(df)} entradas / "
                f"CSV file saved to {arquivo} with {len(df)} entries")
    
    # Log da distribuição das classes / Log class distribution
    contagem = df[SENTIMENT_COLUMN].value_counts().to_dict()
    logger.info("Distribuição das classes / Class distribution:")
    for classe, count in contagem.items():
        logger.info(f"  - {classe}: {count} entradas / entries")

def configurar_acessibilidade_matplotlib() -> None:
    """
    Configura o Matplotlib para gráficos acessíveis.
    Define tamanhos de fonte, tamanho da figura, adiciona grade e usa estilo de alto contraste.
    
    Configure Matplotlib for accessible plots.
    Sets font sizes, figure size, adds grid, and uses a high-contrast style.
    """
    plt.ion()  # Ativa modo interativo / Enable interactive mode
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['axes.grid'] = True
    plt.style.use('seaborn-v0_8-darkgrid')  # Estilo mais acessível / More accessible style
    logger.info("Configurações de acessibilidade do Matplotlib aplicadas / Matplotlib accessibility settings applied")

def salvar_descricao_grafico(dados: pd.DataFrame, caminho_arquivo: str) -> None:
    """
    Salva uma descrição textual do gráfico de distribuição de sentimentos para acessibilidade.
    
    Saves a textual description of the sentiment distribution plot for accessibility.
    
    Args:
        dados (pd.DataFrame): DataFrame com os dados / DataFrame with data
        caminho_arquivo (str): Caminho para salvar / Path to save description
    """
    contagem = dados[SENTIMENT_COLUMN].value_counts().to_dict()
    descricao = "Descrição do Gráfico de Distribuição de Sentimentos:\n"
    descricao += "O gráfico de barras mostra a contagem de cada categoria de sentimento no conjunto de dados.\n"
    for sentimento, count in contagem.items():
        descricao += f"Sentimento '{sentimento}': {count} instâncias.\n"
    descricao += f"Total de instâncias no conjunto de dados: {dados[SENTIMENT_COLUMN].count()}."
    with open(caminho_arquivo, 'w', encoding='utf-8') as f:
        f.write(descricao)
    logger.info(f"Descrição do gráfico salva em {caminho_arquivo} / Plot description saved to {caminho_arquivo}")

def analisar_importancia_features(
    vetorizador: TfidfVectorizer, modelo: LogisticRegression, caminho_arquivo: str, top_n: int = 10
) -> None:
    """
    Analisa os coeficientes do modelo para identificar as features (termos) mais importantes
    e salva os resultados em um arquivo CSV. Também loga os top N termos para acessibilidade.
    
    Analyzes model coefficients to identify the most important features (terms)
    and saves the results to a CSV file. Also logs the top N terms for accessibility.
    
    Args:
        vetorizador (TfidfVectorizer): Vetorizador treinado / Trained vectorizer
        modelo (LogisticRegression): Modelo treinado / Trained model
        caminho_arquivo (str): Caminho para salvar / Path to save results
        top_n (int): Número de features mais importantes a logar / Number of top features to log
    """
    logger.info("Analisando importância das features do modelo... / Analyzing model feature importance...")
    features = vetorizador.get_feature_names_out()
    if modelo.coef_.shape[0] == 1:  # Classificação binária / Binary classification
        coeficientes = modelo.coef_[0]
    else:  # Multiclasse, usar soma absoluta dos coeficientes / Multiclass, use absolute sum of coefficients
        coeficientes = np.sum(np.abs(modelo.coef_), axis=0)
    
    importancia = pd.DataFrame({'Feature': features, 'Importância': coeficientes})
    importancia_ordenada = importancia.sort_values(by='Importância', ascending=False)
    importancia_ordenada.to_csv(caminho_arquivo, index=False, encoding='utf-8')
    logger.info(f"Importância das features salva em {caminho_arquivo} / Feature importance saved to {caminho_arquivo}")
    
    logger.info(f"\n--- Top {top_n} Features Mais Importantes / Top {top_n} Most Important Features ---")
    logger.info("Estes são os termos que mais influenciam as previsões do modelo. / "
                "These are the terms that most influence the model's predictions.")
    for i, row in importancia_ordenada.head(top_n).iterrows():
        logger.info(f"  - '{row['Feature']}': {row['Importância']:.4f}")
    logger.info("------------------------------------------")

def carregar_e_validar_dados(arquivo: str) -> Optional[pd.DataFrame]:
    """
    Carrega e valida dados do CSV. Se o arquivo não existir, gera um CSV sintético com 27 classes.
    
    Load and validate CSV data. If the file doesn't exist, generates a synthetic CSV with 27 classes.
    
    Args:
        arquivo (str): Caminho do arquivo CSV / Path to CSV file
    Returns:
        pd.DataFrame | None: DataFrame válido ou None se houver erro / Valid DataFrame or None if error
    """
    logger.info(f"Verificando existência do arquivo {arquivo} / Checking existence of file {arquivo}")
    if not os.path.exists(arquivo):
        logger.info(f"Arquivo {arquivo} não encontrado. Gerando dados sintéticos... / "
                    f"File {arquivo} not found. Generating synthetic data...")
        gerar_dados_sinteticos(arquivo)
    
    logger.info(f"Carregando dados de {arquivo} / Loading data from {arquivo}")
    try:
        dados = pd.read_csv(arquivo)
        if TEXT_COLUMN not in dados.columns or SENTIMENT_COLUMN not in dados.columns:
            raise ValueError(f"CSV deve conter colunas '{TEXT_COLUMN}' e '{SENTIMENT_COLUMN}' / "
                             f"CSV must contain '{TEXT_COLUMN}' and '{SENTIMENT_COLUMN}' columns")
        initial_rows = len(dados)
        dados.dropna(subset=[TEXT_COLUMN, SENTIMENT_COLUMN], inplace=True)
        if len(dados) < initial_rows:
            logger.warning(f"Removidas {initial_rows - len(dados)} linhas com valores nulos / "
                           f"Removed {initial_rows - len(dados)} rows with null values")
        if dados.empty:
            raise ValueError("DataFrame vazio após remoção de nulos / Empty DataFrame after removing nulls")
        if len(dados) < 500:
            logger.warning(f"O conjunto de dados contém apenas {len(dados)} entradas, o que pode limitar o desempenho do modelo. "
                           f"Recomenda-se pelo menos 1.000 entradas por classe para melhores resultados. / "
                           f"The dataset contains only {len(dados)} entries, which may limit model performance. "
                           f"At least 1,000 entries per class are recommended for better results.")
        logger.info(f"Dados carregados: {len(dados)} registros / Data loaded: {len(dados)} records")
        return dados
    except FileNotFoundError:
        logger.error(f"Arquivo não encontrado: {arquivo} / File not found: {arquivo}")
    except pd.errors.EmptyDataError:
        logger.error(f"Arquivo CSV vazio / Empty CSV file")
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e} / Error loading data: {e}")
    return None

def pre_processar_e_vetorizar(
    texto_treinamento: pd.Series, texto_teste: pd.Series
) -> Tuple[TfidfVectorizer, np.ndarray, np.ndarray]:
    """
    Cria e ajusta um vetorizador TF-IDF para converter texto em recursos numéricos.
    
    Creates and fits a TF-IDF vectorizer to convert text into numerical features.
    
    Args:
        texto_treinamento (pd.Series): Textos para treinamento / Training text series
        texto_teste (pd.Series): Textos para teste / Test text series
    Returns:
        Tuple[TfidfVectorizer, np.ndarray, np.ndarray]: Vetorizador, textos de treino e teste vetorizados /
                                                       Vectorizer, vectorized training and test texts
    """
    logger.info("Iniciando vetorização TF-IDF do texto... / Starting TF-IDF text vectorization...")
    vetorizador = TfidfVectorizer(max_features=MAX_FEATURES_TFIDF, stop_words=None)
    texto_treinamento_vec = vetorizador.fit_transform(texto_treinamento)
    texto_teste_vec = vetorizador.transform(texto_teste)
    logger.info(f"Vetorização concluída. Dicionário TF-IDF com {len(vetorizador.vocabulary_)} termos / "
                f"Vectorization complete. TF-IDF dictionary with {len(vetorizador.vocabulary_)} terms")
    logger.info(f"Dimensões dos dados de treinamento: {texto_treinamento_vec.shape} / "
                f"Dimensions of training data: {texto_treinamento_vec.shape}")
    logger.info(f"Dimensões dos dados de teste: {texto_teste_vec.shape} / "
                f"Dimensions of test data: {texto_teste_vec.shape}")
    return vetorizador, texto_treinamento_vec, texto_teste_vec

def treinar_e_otimizar_modelo(X_train: np.ndarray, y_train: pd.Series) -> LogisticRegression:
    """
    Treina um modelo de Regressão Logística com otimização de hiperparâmetros.
    
    Trains a Logistic Regression model with hyperparameter optimization.
    
    Args:
        X_train (np.ndarray): Dados de treinamento vetorizados / Vectorized training data
        y_train (pd.Series): Rótulos de treinamento / Training labels
    Returns:
        LogisticRegression: Modelo treinado / Trained model
    """
    logger.info("Iniciando treinamento com GridSearchCV / Starting training with GridSearchCV")
    modelo_base = LogisticRegression(max_iter=MAX_ITER_LR, solver='liblinear', random_state=RANDOM_SEED)
    param_grid = {'C': [0.1, 1.0, 10.0], 'penalty': ['l1', 'l2']}
    grid_search = GridSearchCV(modelo_base, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    logger.info(f"Melhores parâmetros: {grid_search.best_params_} / Best parameters: {grid_search.best_params_}")
    logger.info(f"Melhor pontuação (cross-validation): {grid_search.best_score_:.4f} / "
                f"Best score (cross-validation): {grid_search.best_score_:.4f}")
    
    scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    logger.info(f"Acurácia média CV: {np.mean(scores):.2f} ± {np.std(scores):.2f} / "
                f"CV mean accuracy: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    
    logger.info("Modelo treinado com sucesso / Model trained successfully")
    return grid_search.best_estimator_

def avaliar_modelo(modelo: LogisticRegression, X_test: np.ndarray, y_test: pd.Series, classes: list, formato_relatorio: str) -> None:
    """
    Avalia o desempenho do modelo e gera métricas acessíveis em TXT, HTML ou PDF.
    
    Evaluates model performance and generates accessible metrics in TXT, HTML, or PDF.
    
    Args:
        modelo (LogisticRegression): Modelo treinado / Trained model
        X_test (np.ndarray): Dados de teste vetorizados / Vectorized test data
        y_test (pd.Series): Rótulos de teste / Test labels
        classes (list): Lista de classes / List of classes
        formato_relatorio (str): Formato do relatório (txt, html, pdf) / Report format (txt, html, pdf)
    """
    logger.info("Iniciando avaliação do modelo... / Starting model evaluation...")
    previsoes = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, previsoes)
    relatorio_dict = classification_report(y_test, previsoes, target_names=classes, output_dict=True)
    matriz = confusion_matrix(y_test, previsoes, labels=classes)
    
    logger.info(f"\n--- Resultados da Avaliação do Modelo / Model Evaluation Results ---")
    logger.info(f"Acurácia: {acuracia:.4f} / Accuracy: {acuracia:.4f}")
    logger.info("A acurácia representa a proporção de previsões corretas do modelo em relação ao total de previsões. / "
                "Accuracy represents the proportion of correct predictions relative to the total predictions.")
    
    logger.info("\nRelatório de Classificação / Classification Report:")
    logger.info("Este relatório detalha precisão, recall e F1-Score para cada classe de sentimento. / "
                "This report details precision, recall, and F1-Score for each sentiment class.")
    logger.info("Precisão: Proporção de previsões corretas para uma classe específica. / "
                "Precision: Proportion of correct predictions for a specific class.")
    logger.info("Recall: Proporção de instâncias reais de uma classe que foram corretamente previstas. / "
                "Recall: Proportion of actual instances of a class correctly predicted.")
    logger.info("F1-Score: Média harmônica entre precisão e recall. / "
                "F1-Score: Harmonic mean of precision and recall.")
    for class_name, metrics in relatorio_dict.items():
        if isinstance(metrics, dict):
            logger.info(f"  Classe '{class_name}' / Class '{class_name}':")
            logger.info(f"    - Precisão: {metrics['precision']:.4f} / Precision: {metrics['precision']:.4f}")
            logger.info(f"    - Recall: {metrics['recall']:.4f} / Recall: {metrics['recall']:.4f}")
            logger.info(f"    - F1-Score: {metrics['f1-score']:.4f} / F1-Score: {metrics['f1-score']:.4f}")
            logger.info(f"    - Suporte: {metrics['support']} / Support: {metrics['support']}")
        elif class_name in ['accuracy', 'macro avg', 'weighted avg']:
            logger.info(f"  {class_name.replace('_', ' ').title()}: {metrics:.4f}")
    
    logger.info("\nMatriz de Confusão / Confusion Matrix:")
    logger.info("Mostra o número de previsões corretas e incorretas por classe. / "
                "Shows the number of correct and incorrect predictions per class.")
    matriz_df = pd.DataFrame(matriz, index=[f'Real {c}' for c in classes], columns=[f'Previsto {c}' for c in classes])
    logger.info(f"\n{matriz_df.to_string()}")
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            count = matriz[i, j]
            logger.info(f"  {count} amostras de '{true_class}' previstas como '{pred_class}' / "
                        f"{count} samples of '{true_class}' predicted as '{pred_class}'")
    
    # Gera relatório no formato especificado / Generate report in specified format
    relatorio_texto = f"{PROJECT_METADATA['project_name']} - Versão {PROJECT_METADATA['version']}\n"
    relatorio_texto += f"Autor: {PROJECT_METADATA['author']}\n"
    relatorio_texto += f"Site: {PROJECT_METADATA['website']}\n\n"
    relatorio_texto += f"Data da Análise: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    relatorio_texto += f"Acurácia: {acuracia:.4f}\n\nRelatório de Classificação:\n"
    relatorio_texto += classification_report(y_test, previsoes, target_names=classes)
    relatorio_texto += f"\n\nMatriz de Confusão:\n{matriz_df.to_string()}"
    
    if formato_relatorio == 'txt':
        with open(REPORT_TEXT_PATH, 'w', encoding='utf-8') as f:
            f.write(relatorio_texto)
        logger.info(f"Relatório salvo em {REPORT_TEXT_PATH} / Report saved to {REPORT_TEXT_PATH}")
    
    elif formato_relatorio == 'html':
        env = Environment(loader=FileSystemLoader('.'))
        template = env.from_string("""
        <!DOCTYPE html>
        <html lang="pt-BR">
        <head>
            <meta charset="UTF-8">
            <title>{{ project_name }} - Relatório</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                h1, h2 { color: #333; }
            </style>
        </head>
        <body>
            <h1>{{ project_name }} - Versão {{ version }}</h1>
            <p><strong>Autor:</strong> {{ author }}</p>
            <p><strong>Site:</strong> <a href="{{ website }}">{{ website }}</a></p>
            <p><strong>Data da Análise:</strong> {{ data_analise }}</p>
            <h2>Acurácia</h2>
            <p>{{ acuracia }}</p>
            <h2>Relatório de Classificação</h2>
            <pre>{{ relatorio_classificacao }}</pre>
            <h2>Matriz de Confusão</h2>
            <table role="grid">
                <thead>
                    <tr>
                        <th></th>
                        {% for col in colunas %}
                            <th>{{ col }}</th>
                        {% endfor %}
                    </tr>
                </thead>
                <tbody>
                    {% for index, row in matriz.iterrows() %}
                        <tr>
                            <th>{{ index }}</th>
                            {% for value in row %}
                                <td>{{ value }}</td>
                            {% endfor %}
                        </tr>
                    {% endfor %}
                </tbody>
            </table>
        </body>
        </html>
        """)
        html_content = template.render(
            project_name=PROJECT_METADATA['project_name'],
            version=PROJECT_METADATA['version'],
            author=PROJECT_METADATA['author'],
            website=PROJECT_METADATA['website'],
            data_analise=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            acuracia=f"{acuracia:.4f}",
            relatorio_classificacao=classification_report(y_test, previsoes, target_names=classes),
            matriz=matriz_df,
            colunas=matriz_df.columns
        )
        with open(REPORT_HTML_PATH, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Relatório salvo em {REPORT_HTML_PATH} / Report saved to {REPORT_HTML_PATH}")
    
    elif formato_relatorio == 'pdf':
        if pypandoc is None:
            logger.error("Erro: A biblioteca 'pypandoc' não está instalada. Instale com 'pip install pypandoc'.")
            logger.info("Para gerar relatórios em PDF, instale também o Pandoc: https://pandoc.org/installing.html")
            logger.info("Use os formatos 'txt' ou 'html' como alternativas.")
            return
        try:
            latex_content = f"""
            \\documentclass{{article}}
            \\usepackage[utf8]{{inputenc}}
            \\usepackage{{booktabs}}
            \\usepackage{{parskip}}
            \\usepackage{{geometry}}
            \\geometry{{a4paper, margin=1in}}
            \\usepackage{{noto}} % Fonte acessível / Accessible font
            \\begin{{document}}
            \\section*{{{PROJECT_METADATA['project_name']} - Versão {PROJECT_METADATA['version']}}}
            \\textbf{{Autor:}} {PROJECT_METADATA['author']}\\\\
            \\textbf{{Site:}} \\url{{{PROJECT_METADATA['website']}}}\\\\
            \\textbf{{Data da Análise:}} {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\\\
            
            \\section*{{Acurácia}}
            {acuracia:.4f}
            
            \\section*{{Relatório de Classificação}}
            \\begin{{verbatim}}
            {classification_report(y_test, previsoes, target_names=classes)}
            \\end{{verbatim}}
            
            \\section*{{Matriz de Confusão}}
            \\begin{{tabular}}{{l | {'c' * len(classes)}}}
            \\toprule
            & { ' & '.join([f'Previsto {c}' for c in classes]) } \\\\
            \\midrule
            """
            for index, row in matriz_df.iterrows():
                latex_content += f"{index} & {' & '.join(map(str, row.values))} \\\\\n"
            latex_content += """
            \\bottomrule
            \\end{{tabular}}
            \\end{{document}}
            """
            with open('temp_relatorio.tex', 'w', encoding='utf-8') as f:
                f.write(latex_content)
            pypandoc.convert_file('temp_relatorio.tex', 'pdf', outputfile=REPORT_PDF_PATH)
            os.remove('temp_relatorio.tex')
            logger.info(f"Relatório salvo em {REPORT_PDF_PATH} / Report saved to {REPORT_PDF_PATH}")
        except ImportError:
            logger.error("Erro: A biblioteca 'pypandoc' não está instalada. Instale com 'pip install pypandoc'.")
            logger.info("Para gerar relatórios em PDF, instale também o Pandoc: https://pandoc.org/installing.html")
        except RuntimeError as e:
            logger.error(f"Erro ao gerar PDF: A ferramenta Pandoc não foi encontrada. "
                         f"Instale o Pandoc para usar esta funcionalidade: https://pandoc.org/installing.html. Detalhes: {e}")
            logger.info("Use os formatos 'txt' ou 'html' como alternativas.")
        except Exception as e:
            logger.error(f"Erro inesperado ao gerar PDF: {e}")
            logger.info("Use os formatos 'txt' ou 'html' como alternativas.")

def salvar_artefatos_modelo(
    modelo: LogisticRegression, vetorizador: TfidfVectorizer, path_modelo: str, path_vetorizador: str
) -> None:
    """
    Salva o modelo treinado e o vetorizador TF-IDF.
    
    Saves the trained model and TF-IDF vectorizer.
    
    Args:
        modelo (LogisticRegression): Modelo treinado / Trained model
        vetorizador (TfidfVectorizer): Vetorizador treinado / Trained vectorizer
        path_modelo (str): Caminho para salvar modelo / Path to save model
        path_vetorizador (str): Caminho para salvar vetorizador / Path to save vectorizer
    """
    logger.info("Salvando modelo e vetorizador... / Saving model and vectorizer...")
    try:
        joblib.dump(modelo, path_modelo)
        joblib.dump(vetorizador, path_vetorizador)
        logger.info(f"Modelo salvo em: {path_modelo} / Model saved to: {path_modelo}")
        logger.info(f"Vetorizador salvo em: {path_vetorizador} / Vectorizer saved to: {path_vetorizador}")
    except Exception as e:
        logger.error(f"Erro ao salvar artefatos: {e} / Error saving artifacts: {e}")

def carregar_artefatos_modelo(
    path_modelo: str, path_vetorizador: str
) -> Tuple[Optional[LogisticRegression], Optional[TfidfVectorizer]]:
    """
    Carrega um modelo e vetorizador previamente salvos.
    
    Loads a previously saved model and vectorizer.
    
    Args:
        path_modelo (str): Caminho do modelo / Path to model
        path_vetorizador (str): Caminho do vetorizador / Path to vectorizer
    Returns:
        Tuple[Optional[LogisticRegression], Optional[TfidfVectorizer]]: Modelo e vetorizador carregados /
                                                                      Loaded model and vectorizer
    """
    logger.info("Carregando modelo e vetorizador salvos... / Loading saved model and vectorizer...")
    try:
        if os.path.exists(path_modelo) and os.path.exists(path_vetorizador):
            modelo = joblib.load(path_modelo)
            vetorizador = joblib.load(path_vetorizador)
            logger.info("Modelo e vetorizador carregados com sucesso / Model and vectorizer loaded successfully")
            return modelo, vetorizador
        else:
            logger.warning("Arquivos de modelo ou vetorizador não encontrados / Model or vectorizer files not found")
    except Exception as e:
        logger.error(f"Erro ao carregar artefatos: {e} / Error loading artifacts: {e}")
    return None, None

def prever_sentimento(modelo: LogisticRegression, vetorizador: TfidfVectorizer, texto: str) -> str:
    """
    Prevê o sentimento de um texto.
    
    Predicts the sentiment of a text.
    
    Args:
        modelo (LogisticRegression): Modelo treinado / Trained model
        vetorizador (TfidfVectorizer): Vetorizador treinado / Trained vectorizer
        texto (str): Texto para prever / Text to predict
    Returns:
        str: Sentimento previsto / Predicted sentiment
    """
    logger.info(f"Prever sentimento para: '{texto}' / Predicting sentiment for: '{texto}'")
    try:
        texto_vec = vetorizador.transform([texto])
        previsao = modelo.predict(texto_vec)[0]
        probabilidades = modelo.predict_proba(texto_vec)[0]
        classes_ordenadas = modelo.classes_
        prob_str = ", ".join([f"{c}: {p:.2f}" for c, p in zip(classes_ordenadas, probabilidades)])
        logger.info(f"Sentimento previsto: '{previsao}' (Probabilidades: {prob_str}) / "
                    f"Predicted sentiment: '{previsao}' (Probabilities: {prob_str})")
        return previsao
    except Exception as e:
        logger.error(f"Erro ao prever sentimento para '{texto}': {e} / Error predicting sentiment for '{texto}': {e}")
        return "Erro na previsão / Prediction error"

def main(arquivo_dados: str, formato_relatorio: str) -> None:
    """
    Função principal para análise de sentimentos.
    
    Main function for sentiment analysis.
    
    Args:
        arquivo_dados (str): Caminho do arquivo CSV / Path to CSV file
        formato_relatorio (str): Formato do relatório (txt, html, pdf) / Report format (txt, html, pdf)
    Versão: 0.2
    Autor: Eliezer Tavares de Oliveira
    Site: www.eliezertavaresdeoliveira.com
    """
    logger.info(f"--- Iniciando {PROJECT_METADATA['project_name']} (v{PROJECT_METADATA['version']}) ---")
    logger.info(f"Desenvolvido por: {PROJECT_METADATA['author']} | Site: {PROJECT_METADATA['website']}")
    
    configurar_acessibilidade_matplotlib()
    
    dados = carregar_e_validar_dados(arquivo_dados)
    if dados is None:
        logger.critical("Falha no carregamento dos dados / Data loading failed")
        sys.exit(1)
    
    plt.figure()
    ax = sns.countplot(x=dados[SENTIMENT_COLUMN], palette='viridis')
    plt.title("Distribuição de Sentimentos / Sentiment Distribution")
    plt.xlabel(f"{SENTIMENT_COLUMN.capitalize()} / {SENTIMENT_COLUMN.capitalize()}")
    plt.ylabel("Contagem / Count")
    plt.xticks(rotation=45, ha='right')  # Rotaciona rótulos para 27 classes / Rotate labels for 27 classes
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(GRAPH_IMAGE_PATH, dpi=300, bbox_inches='tight')
    salvar_descricao_grafico(dados, GRAPH_DESC_PATH)
    plt.show()
    
    classes_sentimento = sorted(dados[SENTIMENT_COLUMN].unique().tolist())
    logger.info(f"Classes de sentimento: {classes_sentimento} / Sentiment classes: {classes_sentimento}")
    
    texto_treinamento, texto_teste, sentimento_treinamento, sentimento_teste = train_test_split(
        dados[TEXT_COLUMN], dados[SENTIMENT_COLUMN], test_size=TEST_SIZE_RATIO,
        random_state=RANDOM_SEED, stratify=dados[SENTIMENT_COLUMN]
    )
    logger.info(f"Dados divididos: {len(texto_treinamento)} treino, {len(texto_teste)} teste / "
                f"Data split: {len(texto_treinamento)} train, {len(texto_teste)} test")
    
    modelo, vetorizador = carregar_artefatos_modelo(MODEL_SAVE_PATH, VECTORIZER_SAVE_PATH)
    
    if modelo is None or vetorizador is None:
        logger.info("Treinando novo modelo... / Training new model...")
        vetorizador, texto_treinamento_vec, texto_teste_vec = pre_processar_e_vetorizar(
            texto_treinamento, texto_teste
        )
        modelo = treinar_e_otimizar_modelo(texto_treinamento_vec, sentimento_treinamento)
        salvar_artefatos_modelo(modelo, vetorizador, MODEL_SAVE_PATH, VECTORIZER_SAVE_PATH)
    else:
        logger.info("Usando modelo e vetorizador salvos / Using saved model and vectorizer")
        texto_treinamento_vec = vetorizador.transform(texto_treinamento)
        texto_teste_vec = vetorizador.transform(texto_teste)
    
    if modelo and vetorizador:
        avaliar_modelo(modelo, texto_teste_vec, sentimento_teste, classes_sentimento, formato_relatorio)
        analisar_importancia_features(vetorizador, modelo, FEATURE_IMPORTANCE_PATH)
    
        logger.info("\n--- Modo Interativo de Previsão / Interactive Prediction Mode ---")
        logger.info("Digite um texto para prever o sentimento. Digite 'sair' para encerrar / "
                    "Enter a text to predict sentiment. Type 'sair' to exit")
        while True:
            try:
                texto_input = input("Digite seu texto / Enter your text: ")
                if texto_input.lower() == 'sair':
                    logger.info("Encerrando modo interativo / Exiting interactive mode")
                    break
                if not texto_input.strip():
                    logger.warning("Texto vazio. Digite algo válido / Empty text. Enter something valid")
                    continue
                prever_sentimento(modelo, vetorizador, texto_input)
            except KeyboardInterrupt:
                logger.info("Modo interativo interrompido / Interactive mode interrupted")
                break
    else:
        logger.error("Modo interativo indisponível: modelo ou vetorizador não carregados / "
                     "Interactive mode unavailable: model or vectorizer not loaded")
    
    logger.info(f"--- {PROJECT_METADATA['project_name']} (v{PROJECT_METADATA['version']}) Concluído ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"{PROJECT_METADATA['project_name']} - Versão {PROJECT_METADATA['version']} / "
                    f"{PROJECT_METADATA['project_name']} - Version {PROJECT_METADATA['version']}"
    )
    parser.add_argument(
        '--arquivo', type=str, default=DATA_FILE_DEFAULT,
        help=f'Caminho do arquivo CSV. Padrão: "{DATA_FILE_DEFAULT}" / '
             f'Path to CSV file. Default: "{DATA_FILE_DEFAULT}"'
    )
    parser.add_argument(
        '--formato-relatorio', type=str, default='html', choices=['txt', 'html', 'pdf'],
        help='Formato do relatório: txt, html, pdf. Padrão: html / '
             'Report format: txt, html, pdf. Default: html'
    )
    args = parser.parse_args()
    main(args.arquivo, args.formato_relatorio)
