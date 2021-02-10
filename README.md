# PS_Race
Processo Seletivo Race

Foi realizado o exercicio Visão de Camera Lidar, através do Google Colab.
O codigo pode ser visualizado através do Google Colab ao apertar open Colab
Ao apertar no play o coidgo ira pedir para receber o arquivo cvs.

IMPORTANTE 

Para resolver este problema foi-se utilizado a biblioteca sklearng.
Verificando as documentações do sklenarg em seu site ficou claro do que se trata a tecnica de clustering.
As documentações e exemplo encontram-se neste link:

https://scikit-learn.org/stable/modules/clustering.html#clustering

O metódo que estamos utilizando é um aprendizado de maquinas não-supervicionado o que ficou claro pra mim estudando com o link abaixo.

https://www.youtube.com/watch?v=39HBlzFV9vk&t=463s

Estudando o metódo de clustering, descobrir primeiro como funciona o algoritmo K-MEANS, no entanto este não se mostra adequeado para resolver o problema, uma vez que não considera a densidade do problema em seu algortimo se o K-MEANS fosse utilizado a solução seria um grafico dividido em 3 pedaços como uma fatia de pizza.
Então é necessario utilizar o DBSCAN, pois este se prova capaz de verificar a densidade dos pontos com a influência dos vizinhos do lados.

#Construção do Modelo
model = DBSCAN(eps=0.16,min_samples =15,metric='euclidean').fit(dbscan_data)

Para o DBSCAN foi necessario definiar o epsilon do programa, ou seja a distância do raio dos pontos assim o programa irá verificar quanto pontos estão proximos desse raio, já o min_samples está relacionado com a quantidade de pontos que desejo dentro desse epsilon para que ele seja incluindo dentro do clustering e quando estes criterios não são atendidos temos então a condição de outlierns.

Para entender melhor o problema e resolver o problema os seguinte link foram utilizados:

https://www.youtube.com/watch?v=Q7iWANbkFxk 

https://scikit-learn.org/stable/modules/clustering.html#dbscan

Antes de passar pela variavel model, fizemos um pré-processamento dos dados, ou seja o arquivo cvs foi passado para machine learning através destas funções que depois são interpretada como desejamos pela DBSCAN.
#Pré-Processamento
dbscan_data_scaler = StandardScaler().fit(dbscan_data) 
dbscan_data = dbscan_data_scaler.transform(dbscan_data)

Em visualização de resultados encontra-se duas varaveis que estão separando os dados do clustering (inliers) dos outliers.

outliers_df = df[model.labels_==-1] (-1 para o DBSCAN significa outliers)
clusters_df = df[model.labels_!=-1]

Já as proximas linhas serão responsaveis por dar cor aos pontos e assim podemos verificar os dados em através da função plt.figure().
Utilizamos a from collections import Counter, com ela é possivel visualizar a quantidade de clustering gerados em clusters = Counter(model.labels_).

print(clusters)
print('Numero de clusters = {}'.format(len(clusters)-1))

Isso irá gerar um output com os dados de clustering sendo 0,1 e 2 inlierns e -1 como outlierns vem a informação da quantidade de cada um.



