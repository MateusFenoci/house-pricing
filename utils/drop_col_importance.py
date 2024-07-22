import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #Plotar Graficos
import seaborn as sns #Matriz de confusão/histograma
from sklearn.preprocessing import LabelEncoder # Preprocessamento de variaveis categoricas
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.base import clone



seed = 42


train = pd.read_csv('data/train.csv', sep=',')
test = pd.read_csv('data/test.csv', sep=',')


numerical_features = train.select_dtypes(include=['int64', 'float64']).columns
numerical_features = numerical_features.drop('SalePrice')  # Exluindo variavel alvo para que ela nao seja normalizada


# Preenchendo variaveis numericas com a media da coluna
for feature in numerical_features:
    train[feature] = train[feature].fillna(train[feature].median())
    test[feature] = test[feature].fillna(test[feature].median())




# Preenchendo variaveis categoricas com None
categorical_features = train.select_dtypes(include=['object']).columns
for feature in categorical_features:
    train[feature] = train[feature].fillna('None')
    test[feature] = test[feature].fillna('None')


X = train.drop(['SalePrice',"Id"], axis=1) # Exclude the target variable 
y = train['SalePrice']



#Label encoding para variaveis categoricas


#Normalmente temos 79 features, get_dummies eleva esse numero para mais de 300
#Essas novas variaveis criadas geram incerteza na importancia da variavel
label_encoders = {}

for feature in categorical_features:
    le = LabelEncoder()
    X[feature] = le.fit_transform(X[feature])
    label_encoders[feature] = le

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)




## Treinando o modelo
model = RandomForestRegressor(n_estimators=100, random_state=42,oob_score = True)

model.fit(X_train, y_train)

pred = model.predict(X_val)
rmsle = mean_squared_log_error(y_val,pred) ** 0.5

## Avaliando o modelo

print(f"Mean RMSlE: {rmsle}")


#Criar um data frame com a importancia das variaveis
def imp_df(column_names, importances):
    df = pd.DataFrame({'feature': column_names,'feature_importance': importances}).sort_values('feature_importance', ascending = False)
    return df


#Extrair a importancia das variaveis usando drop column
#Artigo : https://explained.ai/rf-importance/
def drop_col_feat_imp(model, X_train, y_train, random_state = 42):
    
    # Clonando um modelo referencia
    model_clone = clone(model)
    model_clone.random_state = random_state
    model_clone.fit(X_train, y_train)
    benchmark_score = model_clone.score(X_train, y_train)
    importances = []

    for col in X_train.columns:
        
        #Treinar o mesmo modelo com uma coluna a menos e avaliar este impacto na performance do modelo
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X_train.drop(col, axis = 1), y_train)
        drop_col_score = model_clone.score(X_train.drop(col, axis = 1), y_train)
        importances.append(benchmark_score - drop_col_score)

        print(f"{col} = {benchmark_score - drop_col_score}")

    importances_df = imp_df(X_train.columns, importances)
    return importances_df



#Plotando um grafico de barras com as importancias das variaveis
importances = drop_col_feat_imp(model,X_train,y_train)
importances.to_csv("importances.csv")
plt.barh(importances["feature"],importances["feature_importance"])
plt.yticks(fontsize=6)
plt.show()


#Plotando um histograma para a variavel alvo
plt.figure(figsize=(9, 8))
plt.title("Frequencia em SalePrice")
sns.distplot(y, color='g', bins=100, hist_kws={'alpha': 0.4})

plt.show()


#Plotando um grafico com a correlação entre as variaveis
data = pd.concat([X_train,y_train],axis=1)
corr_matrix = data.corr()
sns.heatmap(corr_matrix.iloc[-10:, -10:], annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.show()

