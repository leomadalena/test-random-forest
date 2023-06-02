import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump, load

dados = pd.read_csv("data/csv/formatado_animeV1.csv")
# Dados de exemplo para previsão

novos_dados = pd.DataFrame({
    'name': dados["name"],
    'genres': dados["genres"],
    'type': dados["type"],
    'episodes': dados["episodes"],
    'premiered': dados["premiered"],
    'producers': dados["producers"],
    'licensors': dados["licensors"],
    'studios': dados["studios"],
    'source': dados["source"],
    'rating': dados["rating"],
    'popularity': dados["popularity"],
    'favorites': dados["favorites"]
})



# Fazer previsões nos novos dados
#previsoes = pipeline.predict(novos_dados)

# Carregar o modelo a partir do arquivo
modelo_carregado = load('modelo.joblib')

# Usar o modelo carregado para fazer previsões
y_pred = modelo_carregado.predict(novos_dados)


# Exibir as previsões
for i, previsao in enumerate(y_pred):
    print(f"Previsão para o exemplo {i+1}: {previsao}")