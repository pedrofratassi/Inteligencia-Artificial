
# Relátorio do KNN no database do Titanic

Este relatório descreve a aplicação do método KNN (K-Nearest Neighbors) no conjunto de dados do Titanic. O objetivo é construir um modelo preditivo para identificar se um passageiro sobreviveu ou não, com base nas características disponíveis, como classe do passageiro, idade, número de familiares a bordo, entre outros.

<h2>O que é KNN?</h2>
O K-Nearest Neighbors (KNN) é um algoritmo de aprendizado supervisionado amplamente utilizado para tarefas de classificação e regressão. Ele se baseia na ideia de que exemplos semelhantes estão próximos uns dos outros no espaço de características. Para classificar um novo ponto, o KNN verifica os k vizinhos mais próximos ao ponto e atribui a ele a classe mais comum entre esses vizinhos. No caso do Titanic, isso significa que, para prever se um passageiro sobreviveu ou não, o modelo usa os dados de passageiros semelhantes para fazer essa previsão.

O algoritmo KNN não faz suposições sobre a distribuição dos dados, o que o torna muito flexível e fácil de usar. No entanto, ele pode ser sensível à escala das variáveis, o que exige a normalização dos dados para evitar que variáveis com maior magnitude dominem a distância.


## Melhorias

Nesta versão da base de dados do Titanic, foi adicionado o algoritmo KNN, com o objetivo de avaliar sua performance na tarefa de predição de sobrevivência. O modelo foi testado com diferentes valores para o parâmetro k (número de vizinhos), e a eficácia do modelo foi analisada com as principais métricas de avaliação, como a matriz de confusão e a curva ROC.


## Funcionalidades Acrescentadas 

__Carregar os Dados e Definir X e y__<br>
Primeiro passo é carregar os dados e selecionar as variáveis (features) que você deseja usar no modelo. A variável alvo (y) será Survived, que indica se um passageiro sobreviveu ou não.

```markdown
# Definindo os valor para as colunas 'X' e 'y'
X = titanic[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex']]  # Features escolhidas
y = titanic['Survived']  # Variável alvo

# Convertendo as variáveis categóricas em numéricas
X = pd.get_dummies(X, drop_first=True)  # Converte 'Sex' para valores numérico
```

__Legenda:__ 

- __X:__ As features selecionadas para o modelo (ex: classe do passageiro, idade, número de irmãos/cônjuges a bordo, etc.).

- __y:__ A variável que queremos prever (sobrevivência).

- __pd.get_dummies():__ Converte a variável categórica 'Sex' (masculino/feminino) em valores numéricos.

<br>

__Dividir os Dados em Treino e Teste__<br>
Após preparar os dados, eles são divididos em conjuntos de treino e teste. O treino será usado para ajustar o modelo, e o teste para avaliar sua performance.

```markdown
# Dividindo os dados em Treino e Teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

__Explicação:__<br>
- __train_test_split:__ Divide os dados em um conjunto de treino (80%) e teste (20%). O parâmetro random_state=42 garante que a divisão seja sempre a mesma quando você repetir a execução.

<br>

__Normalização dos Dados__<br>
O KNN é sensível à escala das variáveis, então, é importante normalizar as features para que todas fiquem na mesma faixa de valores.

```markdown
# Normalização dos dados
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
__Explicação:__<br>

- __StandardScaler:__ Transforma as features para que tenham média 0 e desvio padrão 1. Isso é importante para evitar que variáveis com magnitudes maiores dominem as distâncias calculadas pelo KNN.

<br>

__Treinar o Modelo KNN__<br>
Aqui cria o modelo de K-Nearest Neighbors e o treina com os dados normalizados.

```markdown
# Treinando o KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)  # Define 5 vizinhos como valor inicial para k
knn.fit(X_train, y_train)
```
__Explicação:__<br>
- __KNeighborsClassifier:__ Este é o algoritmo KNN da scikit-learn. O parâmetro n_neighbors=5 define o número de vizinhos mais próximos que serão usados para classificar um ponto. O valor de k pode ser ajustado conforme necessário

<br>

__Fazer Previsões__<br>
Com o modelo treinado, você pode fazer previsões sobre o conjunto de teste.

```markdown
# Fazendo as previsões
y_pred = knn.predict(X_test)
```

__Explicação:__<br>
- __predict():__ O método predict usa o modelo treinado para prever os rótulos (Survived) no conjunto de teste.

<br>

__Matriz de Confusão__<br>
A matriz de confusão mostra como o modelo está classificando os dados, separando verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos.

```markdown
# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Visualizando a matriz de confusão
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prevista')
plt.ylabel('Real')
plt.title('Matrix de Confusão')
plt.show()
```

__Explicação:__<br>
- __Matriz de Confusão:__ Uma visualização que compara os rótulos reais com as previsões feitas pelo modelo, mostrando os erros e acertos. Isso ajuda a identificar se o modelo está confundindo as classes.

<br>

__Curva ROC e AUC__<br>
A curva ROC mostra a relação entre a taxa de verdadeiros positivos e falsos positivos para diferentes limiares de decisão. O AUC indica a performance geral do modelo.

```markdown
# Prever as probabilidades
y_prob = knn.predict_proba(X_test)[:, 1]

# Calcula a curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Plotando a curva ROC e AUC
plt.plot(fpr, tpr, label=f'KNN (AUC = {roc_auc_score(y_test, y_prob):.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')  # Linha diagonal de referência
plt.xlabel('Taxa de falso positivo')
plt.ylabel('Taxa Verdadeiramente Positiva')
plt.title('Curva ROC')
plt.legend()
plt.show()
```

__Explicação:__<br>
- __Curva ROC:__ Visualiza a capacidade do modelo de distinguir entre classes. Quanto mais próxima a curva estiver do canto superior esquerdo, melhor será a performance do modelo.

- __AUC (Area Under the Curve):__ Mede a área sob a curva ROC; quanto mais próximo de 1, melhor o desempenho.

<br>

__Histograma da Distribuição de Classes__<br>
Esse gráfico mostra a distribuição das classes no conjunto de teste, ajudando a entender se há algum desequilíbrio nas classes.

```markdown
# Histograma da Distribuição de Classes
sns.countplot(x=y_test)
plt.title('Distribuição de Classes no Conjunto de Teste')
plt.show()
```

__Explicação:__<br>
- __Histograma de Distribuição:__ Mostra o número de instâncias de cada classe (0 = não sobreviveu, 1 = sobreviveu), o que é importante para avaliar possíveis desbalanceamentos no dataset.

<br>

__Ajuste de Parâmetros do KNN__<br>
O valor de k (número de vizinhos) pode impactar muito o desempenho do modelo. Experimente variar k e observar como isso afeta os resultados.

```markdown
# Testa diferentes valores de k
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    print(f'k={k}, Test Accuracy: {knn.score(X_test, y_test):.2f}')
```

__Explicação:__<br>
- __Variação de k:__ Testa diferentes valores de k para encontrar o melhor número de vizinhos para o modelo KNN.
## Conclusão

O processo de implementação do KNN no dataset Titanic envolveu várias etapas críticas. Inicialmente, os dados foram preparados e divididos em conjuntos de treino e teste. Em seguida, o modelo KNN foi treinado com as variáveis escolhidas e normalizado para garantir que todas as features tivessem o mesmo peso na análise.

Após treinar o modelo, utilizamos a matriz de confusão para avaliar a eficácia da classificação, observando os acertos e erros. Além disso, a curva ROC nos deu uma visão clara da capacidade do modelo em distinguir entre as classes, com a métrica AUC fornecendo uma avaliação numérica da performance.

Por fim, testamos diferentes valores de k, ajustando o número de vizinhos, e observamos o impacto dessa variação no desempenho do modelo. Essa análise foi essencial para encontrar o valor de k ideal.

Com esse processo, conseguimos construir um modelo de KNN robusto para prever a sobrevivência dos passageiros do Titanic, demonstrando como o KNN pode ser uma técnica eficaz em problemas de classificação binária. No entanto, é importante ressaltar que o desempenho do modelo pode ser melhorado com a otimização de parâmetros e o uso de outras técnicas de pré-processamento.

## Autores

- [@pedrofratassi](https://www.github.com/pedrofratassi)


## Licença

[MIT](https://choosealicense.com/licenses/mit/)

