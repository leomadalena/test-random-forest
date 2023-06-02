import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Carregar o arquivo CSV em um DataFrame do pandas
data = pd.read_csv('data/csv/formatado_animeV1.csv')

# Selecionar os atributos relevantes para treinamento
features = ['name', 'score', 'genres', 'type', 'episodes', 'premiered', 'producers', 'licensors', 'studios', 'source', 'rating', 'popularity', 'favorites']
target = 'score'  # Atributo a ser previsto

# Pré-processamento dos dados
data = data[features]  # Selecionar apenas as colunas relevantes
data = data.dropna()  # Remover linhas com valores ausentes

# Pré-processamento dos atributos numéricos
numeric_features = ['episodes','popularity', 'favorites']
numeric_transformer = StandardScaler()

# Pré-processamento dos atributos categóricos
categorical_features = ['name', 'genres', 'type', 'premiered', 'producers', 'licensors', 'studios', 'source', 'rating']
categorical_transformer = OneHotEncoder(sparse=False, handle_unknown='ignore')

# Combinar os pré-processadores em um ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Dividir os dados em conjunto de treinamento e teste
X = data.drop(target, axis=1)
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajustar o pré-processamento dos atributos categóricos com todas as categorias
categorical_transformer.fit(X[categorical_features])

# Atualizar as categorias conhecidas para o pré-processamento
categorical_transformer.categories_ = [categorical_transformer.categories_[i] for i in range(len(categorical_features))]

# Criar o pipeline com pré-processamento e classificador de árvore de decisão
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor())
])

# Treinar o pipeline
pipeline.fit(X_train, y_train)

# Avaliar o desempenho do pipeline
accuracy = pipeline.score(X_test, y_test)
print("Acurácia:", accuracy)

#até aqui funciona
# Dados de exemplo para previsão
novos_dados = pd.DataFrame({
    'name': ['Chasdlfkjç', 'Isekai de Cheat Skill wo Te ni Shita Ore wa, Genjitsu Sekai wo mo Musou Suru: Level Up wa Jinsei wo Kaeta'],
    'genres': ['Adventure, Comedy, Drama, Sci-Fi, Space', 'Action, Adventure, Fantasy'],
    'type': ['TV', 'TV'],
    'episodes': [26, 13],
    'premiered': ['Springfa 2019', 'Spring 2023'],
    'producers': ['Bandai Visualasdf', 'TMS Entertainment, Tokyo MX, BS11, Kadokawa, CTW'],
    'licensors': ['Funimation, Bandai Entertainment, Sunrise', ''],
    'studios': ['Sunrise', 'Millepensee'],
    'source': ['Manga', 'Light novel'],
    'rating': ['R - 17+ (violence & profanity)', 'PG-13 - Teens 13 or older'],
    'popularity': [39, 1282],
    'favorites': [61971, 1465]
})

# Fazer previsões nos novos dados
previsoes = pipeline.predict(novos_dados)

# Exibir as previsões
for i, previsao in enumerate(previsoes):
    print(f"Previsão para o exemplo {i+1}: {previsao}")