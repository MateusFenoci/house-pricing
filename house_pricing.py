import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import skew


train_data = pd.read_csv('data/train.csv', sep=',')
test_data = pd.read_csv('data/test.csv', sep=',')

##print(train_data.head())
##print(train_data.info())

def show_missing_values(train_data, test_data):
    """ Recebe as databases e printa os valores nulos de cada uma. """
    missing_train = train_data.isnull().sum()
    missin_test = test_data.isnull().sum()
    missing_train = missing_train[missing_train > 0]
    missin_test = missin_test[missin_test > 0]
    missing_train.sort_values(inplace=True)
    missin_test.sort_values(inplace=True)
    print(missing_train)
    print('--------------------')
    print(missin_test)


##show_missing_values(train_data, test_data)


numeric_cols = train_data.select_dtypes(include=[np.number]).columns
numeric_cols = numeric_cols.drop('SalePrice')  # Removendo SalesPrice por ser a variavel alvo e não ter no test_data

## Prenchendo com a média para colunas númericas
imputer = SimpleImputer(strategy='mean')
train_data[numeric_cols] = imputer.fit_transform(train_data[numeric_cols])
test_data[numeric_cols] = imputer.transform(test_data[numeric_cols])

## Preenchendo com a moda para colunas categoricas
categorical_cols = train_data.select_dtypes(include=[object]).columns
imputer = SimpleImputer(strategy='most_frequent')
train_data[categorical_cols] = imputer.fit_transform(train_data[categorical_cols])
test_data[categorical_cols] = imputer.transform(test_data[categorical_cols])

## Checando se ainda temos valores nulos
##show_missing_values(train_data, test_data)


## Transformando variáveis categoricas em númericas
cols = train_data.select_dtypes(include=['object']).columns
for col in cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])

## Normalizando assimetrias para melhorar o desempenho do modelo
numeric_feats = train_data.dtypes[train_data.dtypes != "object"].index
skewed_feats = train_data[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

for feat in skewed_feats:

    ## Separando as features por serem diferentes
    if feat in test_data.columns:
        train_data[feat] = np.log1p(train_data[feat])
        test_data[feat] = np.log1p(test_data[feat])
    else:
        train_data[feat] = np.log1p(train_data[feat])

## Separando teste e treino
X = train_data.drop(['SalePrice', 'Id'], axis=1)
y = train_data['SalePrice']

## Padronizando os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)
test_data_scaled = scaler.transform(test_data.drop('Id', axis=1))

## Treinando o modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

## Avaliando o modelo
scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=10)
rmse_scores = np.sqrt(-scores)
##print(f"Mean RMSE: {rmse_scores.mean()}")

## Fazendo previsões
predictions = model.predict(test_data_scaled)

## Gerando submissão para o desafio Kaggle
submission = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': predictions
})
submission.to_csv('submission.csv', index=False)