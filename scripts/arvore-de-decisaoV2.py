import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump, load



class Main:
    def __init__(self):
        self.data = pd.read_csv("data/csv/formatado_animeV1.csv")
        self.columns = self.data.columns
        self.numeric_features = [i for i in self.data.columns if str(self.data[i].dtype) in 'int64 float64']
        self.categorical_features = [i for i in self.data.columns if str(self.data[i].dtype) == 'object']
        self.target = 'score'

    def treinamento(self):
        self.data = self.columns 
        self.data = self.data.dropna()  
        
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(sparse=False, handle_unknown='ignore')

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
    
        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        categorical_transformer.fit(X[self.categorical_features])

        categorical_transformer.categories_ = [categorical_transformer.categories_[i] for i in range(len(self.categorical_features))]

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', DecisionTreeRegressor())
        ])

        pipeline.fit(X_train, y_train)

        accuracy = pipeline.score(X_test, y_test)
        print("Acurácia:", accuracy)
        
        dump(pipeline, 'modelo.joblib')

        
    
    def teste(self):
        novos_dados = pd.DataFrame({
            'name': ['Chasdlfkjç', 'Isekai de Cheat Skill wo Te ni Shita Ore wa, Genjitsu Sekai wo mo Musou Suru: Level Up wa Jinsei wo Kaeta'],
            'genres': ['Adventure, Comedy, Drama, Sci-Fi, Space', 'Action, Adventure, Fantasy'],
            'type': ['TV', 'TV'],
            'episodes': [13, 22],
            'premiered': ['Springfa 2019', 'Spring 2023'],
            'producers': ['Bandai Visualasdf', 'TMS Entertainment, Tokyo MX, BS11, Kadokawa, CTW'],
            'licensors': ['Funimation, Bandai Entertainment, Sunrise', ''],
            'studios': ['Sunrise', 'Millepensee'],
            'source': ['Manga', 'Light novel'],
            'rating': ['R - 17+ (violence & profanity)', 'PG-13 - Teens 13 or older'],
            'popularity': [39, 1282],
            'favorites': [61971, 1465]
        })
        modelo_carregado = load('modelo.joblib')

        y_pred = modelo_carregado.predict(novos_dados)

        for i, previsao in enumerate(y_pred):
            print(f"Previsão para o exemplo {i+1}: {previsao}")

main = Main()
main.treinamento()
main.teste()