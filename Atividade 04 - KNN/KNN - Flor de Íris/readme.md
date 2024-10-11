
# Relatório do KNN no Dataset da Flor de Íris

Este relatório descreve a aplicação do método KNN (K-Nearest Neighbors) no conjunto de dados da Flor de Íris. O objetivo é construir um modelo preditivo para classificar as espécies de íris (Setosa, Versicolor, ou Virginica) com base nas medidas de sépalas e pétalas.

<h2>O que é KNN?</h2>
O K-Nearest Neighbors (KNN) é um algoritmo de aprendizado supervisionado amplamente utilizado para tarefas de classificação e regressão. Ele baseia-se na ideia de que exemplos semelhantes estão próximos uns dos outros no espaço de características. Para classificar um novo ponto, o KNN verifica os k vizinhos mais próximos ao ponto e atribui a ele a classe mais comum entre esses vizinhos.<br>
<br>
No caso do dataset da Flor de Íris, queremos prever a espécie da flor (Setosa, Versicolor ou Virginica) usando as medidas das sépalas e pétalas. Como o KNN é sensível à escala das variáveis, normalizamos os dados para garantir que todas as variáveis influenciem igualmente o cálculo da distância.

## Melhorias

Nesta versão do modelo de classificação, foi implementado o algoritmo KNN, visando avaliar sua performance na tarefa de classificação multiclasse. O modelo foi testado com diferentes valores de k (número de vizinhos), e a eficácia foi analisada com as principais métricas de avaliação, como a matriz de confusão e a curva ROC para problemas multiclasse.


## Funcionalidades Acrescentadas 

__Carregar os Dados e Definir X e y__<br>
O primeiro passo é carregar o conjunto de dados e selecionar as variáveis (features) que serão utilizadas para o modelo. A variável alvo (y) será a espécie da flor.

```markdown
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data  # Features: medidas de sépalas e pétalas
y = iris.target  # Variável alvo: espécie da flor
```

<br>

__Dividir os Dados em Treino e Teste__<br>
Após preparar os dados, eles são divididos em conjuntos de treino e teste. O conjunto de treino será usado para ajustar o modelo, e o teste para avaliar seu desempenho.

```markdown
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

__Explicação:__<br>
- __train_test_split:__ Divide os dados em um conjunto de treino (80%) e de teste (20%). A opção random_state=42 garante que a divisão seja sempre a mesma ao repetir a execução.

<br>

__Normalização dos Dados__<br>
O KNN é sensível à escala das variáveis, então é importante normalizar os dados para que todas as features fiquem na mesma faixa de valores.

```markdown
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
__Explicação:__<br>

- __StandardScaler:__ Transforma os dados para que cada feature tenha média 0 e desvio padrão 1. Isso é essencial para garantir que todas as variáveis contribuam igualmente para a distância calculada pelo KNN.

<br>

__Treinar o Modelo KNN__<br>
Aqui, criamos o modelo de K-Nearest Neighbors e o treinamos com os dados normalizados.

```markdown
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)  # Usando 5 vizinhos para iniciar o modelo
knn.fit(X_train, y_train)
```
__Explicação:__<br>
- __KNeighborsClassifier:__ O algoritmo KNN da biblioteca scikit-learn. O parâmetro n_neighbors=5 define que serão usados os 5 vizinhos mais próximos para classificar um ponto.

<br>

__Fazer Previsões__<br>
Com o modelo treinado, podemos realizar previsões sobre o conjunto de teste.

```markdown
# Fazendo as previsões
y_pred = knn.predict(X_test)
```

__Explicação:__<br>
- __predict():__ O método usa o modelo treinado para prever as classes (espécies) no conjunto de teste.

<br>

__Matriz de Confusão__<br>
A matriz de confusão mostra como o modelo está classificando os dados, separando verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos.

```markdown
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualizando a matriz de confusão
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão - Flor de Íris')
plt.show()
```

__Explicação:__<br>
- __Matriz de Confusão:__ A matriz de confusão permite visualizar a performance do modelo na classificação de cada uma das três espécies de flores.

<br>

__Curva ROC e AUC__<br>
A curva ROC mostra a relação entre a taxa de verdadeiros positivos e falsos positivos para diferentes limiares de decisão. O AUC indica a performance geral do modelo. Como este é um problema multiclasse, utilizamos o método "one vs rest".

```markdown
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve

# Binarizando as classes para problemas multiclasse
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_prob = knn.predict_proba(X_test)

# Calculando a curva ROC para cada classe
fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
plt.plot(fpr, tpr)
plt.title('Curva ROC - Flor de Íris')
plt.show()

# AUC para classificação multiclasse
roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
```

__Explicação:__<br>
- __label_binarize:__ Transforma as classes em uma forma binária para calcular a ROC para problemas multiclasse.

<br>

__Ajuste de Parâmetros do KNN__<br>
O valor de k (número de vizinhos) pode impactar o desempenho do modelo. Aqui, experimentamos variar k para observar como isso afeta a acurácia.

```markdown
# Testa diferentes valores de k
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    print(f'k={k}, Acurácia no Teste: {knn.score(X_test, y_test):.2f}')
```

__Explicação:__<br>
- __Variação de k:__ Este laço testa diferentes valores de k para encontrar o número ideal de vizinhos que maximize a performance do modelo.

## Conclusão

A aplicação do KNN no dataset da Flor de Íris foi bem-sucedida e envolveu várias etapas críticas. O modelo foi treinado com diferentes valores de k para avaliar seu impacto na performance. Utilizamos a matriz de confusão para avaliar a eficácia da classificação e visualizamos a curva ROC para verificar o desempenho em problemas multiclasse.

O KNN mostrou-se uma técnica robusta para a classificação das espécies de íris, especialmente após a normalização dos dados, o que garantiu uma contribuição equitativa de todas as variáveis no cálculo das distâncias.

## Autores

- [@pedrofratassi](https://www.github.com/pedrofratassi)


## Licença

[MIT](https://choosealicense.com/licenses/mit/)

