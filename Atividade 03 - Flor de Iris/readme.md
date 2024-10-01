
# Relatório de Análise Exploratória de Dados (AED) - Flor Íris

Este relatório descreve a Análise Exploratória de Dados (AED) realizada sobre um banco de dados contendo informações sobre a flor íris.

<p align="center">
    <img src="https://i.imgur.com/WizCNR1_d.png?maxwidth=520&shape=thumb&fidelity=high" alt="flor de íris"/>
</p>


## Estrutura do Banco de Dados

O banco de dados contém 150 entradas, com as seguintes colunas:

- `sepal_length`: comprimento das sépalas
- `sepal_width`: largura das sépalas
- `petal_length`: comprimento das pétalas
- `petal_width`: largura das pétalas
- `species`: espécie da flor

### Tipos de Dados
As quatro primeiras colunas contêm valores numéricos do tipo `float`, enquanto a coluna `species` contém valores do tipo `object`, representando as espécies das flores. Não há valores ausentes no conjunto de dados.

## Análise de Espécies

Utilizando o método `iris.species.unique()`, foram identificadas as espécies presentes no conjunto de dados.

## Análise Estatística

A função `iris.describe()` foi utilizada para obter uma visão geral das variáveis numéricas. Observamos que:

- O comprimento das sépalas e das pétalas é, em média, maior que suas respectivas larguras.
- O desvio padrão das pétalas é maior do que o das sépalas, indicando maior variação no tamanho das pétalas.

### Boxplot

O gráfico Boxplot revelou a presença de **outliers** na largura das sépalas. Esses pontos podem não ser necessariamente outliers, mas indicam valores fora do esperado para a amostra. Uma investigação mais aprofundada seria necessária para decidir se esses dados devem ser mantidos.

Além disso, o Boxplot destacou a maior dispersão dos dados de comprimento das pétalas, confirmando as observações anteriores.

### Histograma

O histograma foi utilizado para visualizar a distribuição dos dados. Constatamos que:

- Apenas a largura das sépalas apresenta uma distribuição próxima da normal.
- O comprimento das sépalas também pode ter uma distribuição normal, embora essa afirmação não possa ser confirmada apenas visualmente.
- Os dados das pétalas sugerem a existência de pelo menos dois grupos distintos de dados, com pétalas menores e maiores.

## Teste de Normalidade de Shapiro-Wilk

Para confirmar a normalidade dos dados, foi aplicado o teste de Shapiro-Wilk. O resultado indicou que os comprimentos das sépalas **não seguem uma distribuição normal**.

## Gráfico de Dispersão

Utilizando gráficos de dispersão, analisamos a correlação entre o comprimento e a largura das sépalas. Um ajuste linear foi feito utilizando a função `linregress` da biblioteca `scipy.stats`, permitindo verificar a relação entre essas variáveis.

### Regressão Linear

Os principais coeficientes da regressão linear foram:

- **Slope**: coeficiente angular da reta
- **Intercept**: intercepto da reta
- **Rvalue**: valor do coeficiente de correlação de Pearson
- **Pvalue**: valor de p do ajuste
- **Stderr**: erro padrão do modelo
- **Intercept_stderr**: erro padrão do intercepto estimado

## Conclusão

A análise exploratória realizada identificou várias características importantes no conjunto de dados da flor íris, incluindo padrões de distribuição e correlações entre variáveis. A identificação de outliers e a falta de normalidade em algumas variáveis são pontos importantes que podem ser investigados mais profundamente em análises futuras.




## Referência

 - [Database](https://archive.ics.uci.edu/dataset/53/iris)


## Licença

[MIT](https://choosealicense.com/licenses/mit/)

