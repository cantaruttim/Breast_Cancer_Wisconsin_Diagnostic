import pandas as pd
import seaborn as sns
import numpy as np

dfC = pd.read_csv('breast-cancer-wisconsin.data', header=None)
display(dfC)

# Precisamos renomear as colunas, de acordo com a descrição do DataSet
# 
dfC = dfC.rename(columns={0:'Sample code number',
                          1:'Clump Thickness',
                          2:'Uniformity of Cell Size',
                          3:'Uniformity of Cell Shape',
                          4:'Marginal Adhesion',
                          5:'Single Epithelial Cell Size',
                          6:'Bare Nuclei',
                          7:'Bland Chromatin',
                          8:'Normal Nucleoli',
                          9:'Mitoses',
                          10:'Class'})

dfC

dfC.drop(columns=['Sample code number'])

# observamos que mesmo com os valores não nulos, temos '?' como máximo

dfC['Bare Nuclei'].min(), dfC['Bare Nuclei'].max()

# Não temos nenhum valor faltando, porem temos 16 que estão com os valores '?' e devem ser tratados
dfC.isnull().sum()

# observando apenas os indexes com '?' 

dfC2 = dfC.drop(dfC[dfC['Bare Nuclei'] == '?'].index)
dfC2

# Substituindo os 16 valores por 5 (média)

dfC.loc[dfC['Bare Nuclei'] == '?', 'Bare Nuclei'] = 5.0

# Divisão da base

X = dfC.iloc[:, 1:10].values
X

y = dfC.iloc[:, 10].values
y


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix

# Divisão dos dados em treino e teste
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                 test_size=0.25,
                                                                 random_state=0)


from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC



# 95.42%
naive = GaussianNB()
naive.fit(X_treinamento, y_treinamento)
previsoes = naive.predict(X_teste)

cm = ConfusionMatrix(naive)
cm.fit(X_treinamento, y_treinamento)
cm.score(X_teste, y_teste)


# 94.28%
arvore = DecisionTreeClassifier(criterion='entropy')
arvore.fit(X_treinamento, y_treinamento)

cm = ConfusionMatrix(arvore)
cm.fit(X_treinamento, y_treinamento)
cm.score(X_teste, y_teste)

# 97.14%


random_forest = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state = 0)
random_forest.fit(X_treinamento, y_treinamento)

previsoes = random_forest.predict(X_teste)
previsoes

accuracy_score(y_teste, previsoes)

cm = ConfusionMatrix(random_forest)
cm.fit(X_treinamento, y_treinamento)
cm.score(X_teste, y_teste)

# 97.77%

knn_bc = KNeighborsClassifier(n_neighbors= 5, metric='minkowski', p = 2)
knn_bc.fit(X_treinamento, y_treinamento)

previsoes = knn_bc.predict(X_teste)
previsoes

cm = ConfusionMatrix(knn_bc)
cm.fit(X_treinamento, y_treinamento)
cm.score(X_teste, y_teste)

print(classification_report(y_teste, previsoes))


# 96.57%
logistic_bc = LogisticRegression(random_state = 1)
logistic_bc.fit(X_treinamento, y_treinamento)

previsoes = logistic_bc.predict(X_teste)

cm = ConfusionMatrix(logistic_bc)
cm.fit(X_treinamento, y_treinamento)
cm.score(X_teste, y_teste)


# 96.57%
svm_bc = SVC(kernel='rbf', random_state=1, C = 2.0) 
svm_bc.fit(X_treinamento, y_treinamento)

previsoes = svm_bc.predict(X_teste)

cm = ConfusionMatrix(svm_bc)
cm.fit(X_treinamento, y_treinamento)
cm.score(X_teste, y_teste)


## Observamos como o KNN obteve o maior valor de acurácia. Vamos agora aplicar o kFold em ## conjunto com o GridSearchCV e observar o resultado novamente

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

X, y


# aplicando o GridSearch

# Árvore de Decisão
parametros = {'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random'],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10]}

# aplicando o GridSearchCV

grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=parametros)
grid_search.fit(X, y)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(melhores_parametros)
print(melhor_resultado)


# Random Florest

parametros = {'criterion': ['gini', 'entropy'],
              'n_estimators': [10, 40, 100, 150],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10]}


# aplicando o GridSearchCV

grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parametros)
grid_search.fit(X, y)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(melhores_parametros)
print(melhor_resultado)


# KNN

parametros = {'n_neighbors': [3, 5, 10, 20],
              'p': [1, 2]}


grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parametros)
grid_search.fit(X, y)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(melhores_parametros)
print(melhor_resultado)



# Regressão logística

parametros = {'tol': [0.0001, 0.00001, 0.000001],
              'C': [1.0, 1.5, 2.0],
              'solver': ['lbfgs', 'sag', 'saga']}


grid_search = GridSearchCV(estimator=LogisticRegression(), param_grid=parametros)
grid_search.fit(X, y)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_


print(melhores_parametros)
print(melhor_resultado)



# SVC

parametros = {'tol': [0.001, 0.0001, 0.00001],
              'C': [1.0, 1.5, 2.0],
              'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}


grid_search = GridSearchCV(estimator=SVC(), param_grid=parametros)
grid_search.fit(X, y)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(melhores_parametros)
print(melhor_resultado)



# Redes Neurais

parametros = {'activation': ['relu', 'logistic', 'tahn'],
              'solver': ['adam', 'sgd'],
              'batch_size': [10, 56]}



grid_search = GridSearchCV(estimator=MLPClassifier(), param_grid=parametros)
grid_search.fit(X, y)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_




print(melhores_parametros)
print(melhor_resultado)





# kFold

from sklearn.model_selection import cross_val_score, KFold


resultados_arvore = []
resultados_random_forest = []
resultados_knn = []
resultados_logistica = []
resultados_svm = []
resultados_rede_neural = []

for i in range(30):
  print(i)
  kfold = KFold(n_splits=10, shuffle=True, random_state=i)

  arvore = DecisionTreeClassifier(criterion='gini', min_samples_leaf=5, min_samples_split=10, splitter='best')
  scores = cross_val_score(arvore,  X, y, cv = kfold)
  #print(scores)
  #print(scores.mean())
  resultados_arvore.append(scores.mean())

  random_forest = RandomForestClassifier(criterion = 'gini', min_samples_leaf = 1, min_samples_split=5, n_estimators = 10)
  scores = cross_val_score(random_forest,  X, y, cv = kfold)
  resultados_random_forest.append(scores.mean())

  knn = KNeighborsClassifier(n_neighbors=3, p=1)
  scores = cross_val_score(knn,  X, y, cv = kfold)
  resultados_knn.append(scores.mean())

  logistica = LogisticRegression(C = 1.0, solver = 'lbfgs', tol = 0.0001)
  scores = cross_val_score(logistica,  X, y, cv = kfold)
  resultados_logistica.append(scores.mean())

  svm = SVC(kernel = 'linear', C = 1.0)
  scores = cross_val_score(svm,  X, y, cv = kfold)
  resultados_svm.append(scores.mean())

  rede_neural = MLPClassifier(activation = 'relu', batch_size = 10, solver = 'sgd')
  scores = cross_val_score(rede_neural, X, y, cv = kfold)
  resultados_rede_neural.append(scores.mean())




resultados = pd.DataFrame({'Arvore': resultados_arvore, 'Random forest': resultados_random_forest,
                           'KNN': resultados_knn, 'Logistica': resultados_logistica,
                           'SVM': resultados_svm, 'Rede neural': resultados_rede_neural})
resultados



resultados.describe()




# Coeficiente de Variação 

CV = (resultados.std() / resultados.mean()) * 100
CV



from scipy.stats import shapiro # biblioteca que informa se os dados seguem uma distribuição normal ou não
import seaborn as sns


shapiro(resultados_arvore), shapiro(resultados_random_forest), shapiro(resultados_knn), shapiro(resultados_logistica), shapiro(resultados_svm), shapiro(resultados_rede_neural)



from scipy.stats import f_oneway


_, p = f_oneway(resultados_arvore, resultados_random_forest, resultados_knn, resultados_logistica, resultados_svm, resultados_rede_neural)
p


alpha = 0.05
if p <= alpha:
  print('Hipótese nula rejeitada. Dados são diferentes')
else:
  print('Hipótese alternativa rejeitada. Resultados são iguais')



# como os resultados são diferentes, precisamos avaliar realmente qual algoritmo é melhor

resultados_algoritmos = {'accuracy': np.concatenate([resultados_random_forest, resultados_knn, resultados_svm, resultados_rede_neural]),
                         'algoritmo': ['random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest', 
                          'knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn', 
                          'svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm',
                          'rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural']}




resultados_df = pd.DataFrame(resultados_algoritmos)
resultados_df



from statsmodels.stats.multicomp import MultiComparison


compara_algoritmos = MultiComparison(resultados_df['accuracy'], resultados_df['algoritmo'])

teste_estatistico = compara_algoritmos.tukeyhsd()
print(teste_estatistico)


teste_estatistico.plot_simultaneous();
