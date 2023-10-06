#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 18:09:56 2021

@author: paul

IA - Classification Challenge
"""
import pandas as pd 
from math import sqrt
import time
import matplotlib.pyplot as plt

t0 = time.time()

# %% On récupère les dataframe grâce à pandas 

data_learn = pd.read_csv('data.csv',header = None)
data_test = pd.read_csv('pretest.csv',header = None)
#data_learn2 = data_learn.append(data_test,sort=False) # data.csv + pretest.csv
data_testFinal = pd.read_csv('finalTest.csv',header = None)



# %% Séparation de data en deux parties
"""
Sépare la dataframe data en deux sous data frames TRAIN et TEST en choisissant le pourcentage de données dans TRAIN par rapport à DATA
Dans ce code, j'ai choisi n = 0.8 tout le temps mais j'ai parfois testé n pour 0.9 ou 0.5 pour diminuer le temps d'éxécution du code.'
"""
def split(data,n):
    taille_pli = int(len(data)*n)
    df1 = data.iloc[:taille_pli,:]
    df2 = data.iloc[taille_pli:,:]
    return df1,df2

# %% Distance euclidienne entre deux vecteurs
"""
Méthode qui récupère la distance euclidienne entre deux vecteurs.

La première étape est de calculer la distance euclidienne entre deux points (ce qui correspond aux deux lignes row1 
et row2 dans notre dataframe).
Plus la distance est faible, plus les points sont similaires donc sont probablement de même label.
Une distance de 0 signifie que les points sont superposés.
Pour chaque ligne, on ne prend pas en compte la dernière colonne car celle ci représente les labels des points.

On peut noter qu'on aurait pu utiliser la distance Manhattan mais celle ci prend plus de temps d'éxécution après l'avoir 
testé (6:48 contre 5:41 pour finalTest).
"""
def distance_euclidienne(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

def distance_manhattan(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance+=abs(row1[i]-row2[i])
    return distance

# %% Récupérer les k voisins les plus proches
"""
Méthode qui récupère les k voisins les plus proches.

Les voisins d'un point sont ses k plus proches points selon la distance euclidienne définie dans la section précédente.
On calcule la distance euclidienne entre le point à tester (test_row) et tous les points de train (la dataframe qui nous
permet de nous entrainer).
Une fois toutes les distances calculées et introduites dans une liste 'distances', on les range par ordre croissant pour 
récupérer les k premiers voisins.

On return la liste voisins.
"""

def get_voisins(train, test_row, k): 
	distances = []
	for train_row in train.iloc: #On parcourt les points de notre dataframe train
		distance = distance_euclidienne(test_row, train_row) #On calcule la distance euclidienne entre le point à tester test_row et tous les points de train 
		distances.append((train_row, distance))
	distances.sort(key=lambda t: t[1])  #On range les distances par ordre croissant
	voisins = []
	for i in range(0,k):
		voisins.append(distances[i][0])  #On prend les k premiers voisins (point) que l'on insère dans une liste voisins
	return voisins


# %% Analyse des voisins 
"""
Méthode qui crée une prédiction du label d'un point de test à partir du label de ses voisins. La prédiction sera le label le plus présent
parmi les k plus proches voisins du point.
On s'aide donc du dataset d'apprentissage train car on connait les labels.
"""

classes = ['classA','classB','classC','classD','classE']

def analyse_voisins(train,test_row,k):
    occ=[] #occ = occurence pour chaque classe 
    voisins = get_voisins(train,test_row,k) #Récup la liste des k voisins de test_row
    list_label = [a[6] for a in voisins] #Recup la liste des labels des voisinns
    for classe in classes:
        occ.append(list_label.count(classe))  #Remplit la liste des occurences => le nombre de fois chaque classe est présente parmis les voisins
    prediction = classes[occ.index(max(occ))] #récupère la classe la mieux représentée parmis la liste classes avec la fonction max
    return prediction

# %% Algorithme Knn
"""
Méthode qui va créer la liste des prédictions pour la dataframe test dont on ne connait pas les labels à l'aide du dataset d'apprentissage.
C'est en soit la méthode analyse_voisins appliquée pour chaque point de test.
"""

def Knn(train, test, k):
	predictions = list()
	for row in test.iloc:
		prediction = analyse_voisins(train, row, k)
		predictions.append(prediction)
	return(predictions)

# %% Pourcentage de précision des prédictions:
"""
Méthode qui récupère la précision des prédictions d'un data set d'apprentissage.
C'est un counter qui prend +1 pour chaque similarité entre la prédiction et le label du point de data
"""
def get_precision(data,predictions):
    a=0
    count=0
    for prediction in predictions: #parcourt les prédictions
        if prediction == data.iloc[a][6]: #Compare la prédiction à la classe actuelle du point
            count+=1
        a+=1
    return round(count / float(len(data))*100,2) #arrondit à 2 chiffres après la virgule

# %% Evaluation des k 
"""
Méthode qui évalue chaque la précision des prédictions pour chaque k dans un intervalle que l'on choisit.
Pour cela on utilise la méthode Knn et la méthode get_precision pour chaque k de l'intervalle.

Cette méthode nous permet de déterminer quel k choisir pour le test final.
"""
k_range = range(25,101)

def evaluation_des_k(train,test,k_range):
    for k in k_range:
        predictions = Knn(train,test,k)
        a = "précision pour k = {} : {}%".format(k,get_precision(test,predictions))
        print(a)
 
        

"""
evaluation_des_k(data_learn,data_test,k_range)


Résultats pour TRAIN = data.csv et TEST = preset.csv : 
    précision pour k = 1 : 68.24%
    précision pour k = 2 : 61.02%
    précision pour k = 3 : 68.99%
    précision pour k = 4 : 68.49%
    précision pour k = 5 : 69.24%
    précision pour k = 6 : 69.49%
    précision pour k = 7 : 70.49%
    précision pour k = 8 : 70.61%
    précision pour k = 9 : 71.98%
    précision pour k = 10 : 72.1%
    précision pour k = 11 : 72.85%
    précision pour k = 12 : 72.85%
    précision pour k = 13 : 73.85%
    précision pour k = 14 : 72.35%
    précision pour k = 15 : 73.6%
    précision pour k = 16 : 72.48%
    précision pour k = 17 : 73.6%
    précision pour k = 18 : 72.98%
    précision pour k = 19 : 73.1%
    précision pour k = 20 : 72.85%
    précision pour k = 21 : 73.1%
    précision pour k = 22 : 72.6%
    précision pour k = 23 : 73.47%
    précision pour k = 24 : 73.35%
    précision pour k = 25 : 73.6%
temps d'éxécution : 47:12

On remarque bien que le meilleur k ici est 13 -> 73,85 %



On va donc le tester pour les autres datas pour vérifier sa qualité.

Avec data.csv que l'on split en deux parties (80% train, 20% test): 
train,test = split(data_learn,0.8)
evaluation_des_k(train,test,k_range)

    précision pour k = 1 : 91.93%
    précision pour k = 2 : 87.58%
    précision pour k = 3 : 90.68%
    précision pour k = 4 : 91.93%
    précision pour k = 5 : 92.55%
    précision pour k = 6 : 93.79%
    précision pour k = 7 : 93.17%
    précision pour k = 8 : 93.17%
    précision pour k = 9 : 93.17%
    précision pour k = 10 : 91.93%
    précision pour k = 11 : 91.93%
    précision pour k = 12 : 91.3%
    précision pour k = 13 : 90.06%
    précision pour k = 14 : 91.93%
    précision pour k = 15 : 90.06%
    précision pour k = 16 : 90.68%
    précision pour k = 17 : 88.82%
    précision pour k = 18 : 89.44%
    précision pour k = 19 : 89.44%
    précision pour k = 20 : 90.06%
    précision pour k = 21 : 89.44%
    précision pour k = 22 : 90.06%
    précision pour k = 23 : 87.58%
    précision pour k = 24 : 87.58%
    précision pour k = 25 : 86.96%
temps d'exécution: 6:37

On remarque bien que le meilleur k ici est 6 -> 93,79 %
    

Avec pretest.csv que l'on split en deux parties (80% train, 20% test): 
train,test = split(data_test,0.8)
evaluation_des_k(train,test,k_range)

    précision pour k = 1 : 83.85%
    précision pour k = 2 : 81.99%
    précision pour k = 3 : 90.68%
    précision pour k = 4 : 90.06%
    précision pour k = 5 : 90.68%
    précision pour k = 6 : 91.3%
    précision pour k = 7 : 89.44%
    précision pour k = 8 : 92.55%
    précision pour k = 9 : 91.3%
    précision pour k = 10 : 91.3%
    précision pour k = 11 : 88.82%
    précision pour k = 12 : 87.58%
    précision pour k = 13 : 87.58%
    précision pour k = 14 : 88.2%
    précision pour k = 15 : 87.58%
    précision pour k = 16 : 86.96%
    précision pour k = 17 : 85.71%
    précision pour k = 18 : 86.96%
    précision pour k = 19 : 85.09%
    précision pour k = 20 : 85.09%
    précision pour k = 21 : 84.47%
    précision pour k = 22 : 83.85%
    précision pour k = 23 : 83.85%
    précision pour k = 24 : 84.47%
    précision pour k = 25 : 83.85%
temps d'exécution: 6:41

On remarque bien que le meilleur k ici est 8 -> 92,55 %


"""
#Pour un k testé avec len(data)=802 , temps = 1m37s

#train,test = split(data_test,0.8)
#train2,test2 = split(data_learn,0.8)

#evaluation_des_k(train,test,k_range)


# %% Ecriture de la réponse:
"""
Méthode qui va créer un fichier 'COIFFET_labels.txt' qui comprendra toutes les prédictions.
"""
def EcrireLabels(predictions):
    f = open('COIFFET_labels3.txt', 'w')
    for prediction in predictions:
        f.write(prediction)
        f.write("\n")
    f.close()


# %% CheckLabel

allLabels = ['classA','classB','classC','classD','classE']
def Check_label(nameFile,nbLines):
    fd =open(nameFile,'r')
    lines = fd.readlines()
    count=0
    for label in lines:
    	if label.strip() in allLabels:
    		count+=1
    	else:
    		if count<nbLines:
    			print("Wrong label line:"+str(count+1))
    			break
    if count<nbLines:
    	print("Labels Check : fail!")
    else:
    	print("Labels Check : Successfull!")

# %% PLOTS
#Récupère tous les plots possibles par duo de coordonnées pour une dataframe
def plot(filename):
    for i in range(0,6):
        for j in range(0,6):
            if i != j:
                x = filename.loc[:,i]
                y = filename.loc[:,j]
                lab = filename.loc[:,'Labels']
                plt.axis('equal')
                plt.scatter(x[lab == 'classA'], y[lab == 'classA'], color='g', label='classA')
                plt.scatter(x[lab == 'classB'], y[lab == 'classB'], color='c', label='classB')
                plt.scatter(x[lab == 'classC'], y[lab == 'classC'], color='r', label='classC')
                plt.scatter(x[lab == 'classD'], y[lab == 'classD'], color='b', label='classD')
                plt.scatter(x[lab == 'classE'], y[lab == 'classE'], color='y', label='classE')
                plt.title("Nuage de points de data.csv entre coordonnnées {} et {}".format(x,y))
                plt.legend()
                plt.show()
                

# %% ZONE DE TEST

#Main pour écrire les réponses dans le fichier txt
predictions = Knn(data_test,data_testFinal,8)      
EcrireLabels(predictions)
Check_label('COIFFET_sample.txt',3000)

"""
print("Occurence pour chaque classe: ")
print("Classe A :",predictions.count("classA"))
print("Classe B :",predictions.count("classB"))
print("Classe C :",predictions.count("classC"))
print("Classe D :",predictions.count("classD"))
print("Classe E :",predictions.count("classE"))


new_finaltest = data_testFinal.assign(Labels = predictions)

plot(new_finaltest)

"""


# %% Temps d'exécution
t1=time.time()
duree=t1-t0
minutes=int(duree//60)
secondes = round(duree%60)

print("temps d'exécution: {}:{}".format(minutes,secondes))

