
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Sat Mar 20 17:26:15 2021

@author: paul
"""
import numpy as np
import random
import math
import time
# %% CLASSE TRIPLET
start_time = time.time()
# On définit une classe Triplet regroupant le triplet (a,b,c) et le coût associé à ce triplet
class Triplet():
    def __init__(self,a,b,c):
        self.a = a
        self.b = b
        self.c = c
        self.cout=self.Cout()
    def Affichage(self):
        print("({},{},{}):{}".format(self.a,self.b,self.c,self.cout))
       
#Fonction de calcul de coût pour un triplet
    def Cout(self):
        cout=0
        file = np.loadtxt('temperature_sample.csv',delimiter = ";", skiprows=1)
        for i in range(len([row[0] for row in file])):
            somme=0
            for n in range(self.c+1):
                somme+=(self.a**n) * math.cos((self.b**n)*math.pi*file[i][0])
            cout+=abs(somme-file[i][1])
        return cout
    
# %% FONCTIONS   

#On génère n individus aléatoirement ce qui nous donne la base de notre population d'individus
def NIndividu(n):
    i=0
    while(i<n):
        a = round(random.uniform(0,1),3)
        b = round(random.randint(1,20),3)
        c = round(random.randint(1,20),3)
        individu = Triplet(a,b,c)
        listIndividus.append(individu)
        i+=1

#Donne 3n nouveaux individus par croisement génétique
def NCroisement2(n):
    i = 0
    while(i<n):
        r1=random.randint(0, len(listIndividus) - 1)   
        r2=random.randint(0, len(listIndividus) - 1)
        r3=random.randint(0, len(listIndividus) - 1)
        f1=Triplet(listIndividus[r1].a, listIndividus[r2].b, listIndividus[r3].c)
        f2=Triplet(listIndividus[r2].a, listIndividus[r3].b, listIndividus[r1].c)
        f3=Triplet(listIndividus[r3].a, listIndividus[r1].b, listIndividus[r2].c)
        listIndividus.append(f1)
        listIndividus.append(f2)
        listIndividus.append(f3)
        i+=1

#Donne n nouveaux individus par mutation génétique
def NMutation(n):
    i=0
    while(i<n):
        r = random.randint(0,len(listIndividus)-1)
        f1 = listIndividus[r]
        f2 = Triplet(0,0,0)
        f2.a=f1.a
        f2.b=f1.c  
        f2.c=f1.b
        listIndividus.append(f2)
        i+=1

#Fonction qui d'une part actualise le cout de tous les individus puis garde les 20 meilleurs individus dans une nouvelles liste    
def CompareCout2():
    for i in range(20):
        minimum = i
        for j in range(i + 1, len(listIndividus)):
            if(listIndividus[minimum].cout>listIndividus[j].cout):
                minimum = j 
        temp = listIndividus[minimum]
        listIndividus[minimum] = listIndividus[i]
        listIndividus[i] = temp
    del(listIndividus[20::])
    
    
# %% MAIN  
listIndividus=[]
listFitness=[]
NIndividu(100) #Création population de 100 individus de base 
CompareCout2()
#On teste le code sur 30 générations pour avoir une précision suffisante
i=0
while(i<30):
    NCroisement2(30)
    NMutation(100)
    CompareCout2()
    i+=1

#Notre solution :
print(listIndividus[0].Affichage())
#On trouve environ (0.13,19,c)
#Notre b est correcte, a est relativement proche mais cependant c varie énormement
print("--- %s seconds ---" % (time.time() - start_time))