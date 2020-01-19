import pandas as pd
import numpy as np

# useremo il dataset shirt con un elenco di magliette con annesse
# proprietà come taglia colore e prezzo
shirt = pd.read_csv("/Users/massimilianoguida/Repo_Corso_DeepLearning/2 - Datasets e data preprocessing/data/shirts.csv"
                    , index_col=0)

#test per vedere se il path è stato preso correttamente
print(shirt.head())

# creiamo un array numpy contenente il dataset dei primi 10 valori del dataset
X = shirt.values
print(X[:10])

# 0     S     bianco      4.99
# 1     M     bianco      19.99
# 2     XL    bianco      12.49
# 3     XL    bianco      14.99
# 4     S     bianco      14.99
# analizzando il dataset possiamo notare che la TAGLIA è una variabile
# categorica ordinale in quanto è possibile stabilire una relazione di ordine
# tra i vari lable. Mentre colore rappresenta una categorica nominale in quanto
# non è possibile stabilire una relazione d'ordine ma solo di ugualianza
# Vedremo ora 2 metodi che ci permettono di trasformare questi 2 tipi di
# variabili in numeri
#   1. TAGLIA -> ORDINALI
#           è impossibile stabilire un ordine via codice,
#           quindi sarà necessario costruire un dizionario
size_mapping = {"S": 0, "M": 1, "L": 2, "XL": 3}
#           ora usando PANDAS per sostituire il valore contenuto
#           a partire dalla chiave mediante il metodo MAP
shirt["taglia"] = shirt["taglia"].map(size_mapping)
print(shirt.head())

#      taglia  colore  prezzo
# 0       0    bianco    4.99
# 1       1    bianco   19.99
# 2       3    bianco   12.49
# 3       3    bianco   14.99
# 4       0    bianco   14.99
# se il nostro dataset è un array numpy
# e non possiamo usare pandas allora sarà
# necessario implementare la nostra funzione map
# per risparmiarci di costruire una funzione con vari cicli annidati
# possiamo servirci del metodo vectorize all'interno della quale
# creeremo la funzione lambda di associazione
fmap = np.vectorize(lambda t: size_mapping[t])
# ora applichiamo la funzione fmap appena creata sulla
# colonna dell'array che si riferisce alla mappa (che sarebbe la prima)
X[:, 0] = fmap(X[:, 0])
print(X[:5])

#   2. COLORE -> NOMINALI
#           si userà l'algoritmo One-Hot encoding
#           si convertiranno i dati relativi al colore in colonne
#           si useranno valori booleani per item che rappresenteranno
#           l'appartenenza o meno dell'item a quella categoria (bianco, verde o altro)
#           si usa la funzione get_dummies di pandas
shirt = pd.get_dummies(shirt, columns=["colore"])
print(shirt.head())
#    taglia  prezzo  colore_bianco  colore_rosso  colore_verde
# 0       0    4.99              1             0             0
# 1       1   19.99              1             0             0
# 2       3   12.49              1             0             0
# 3       3   14.99              1             0             0
# 4       0   14.99              1             0             0

# Se il nostro dataset è un array numpy possiamo utilizzare la classe OneHotEncoder di scikit-learn.
# Per eseguire il One Hot Encoding su solamente una o più colonne del dataset possiamo sfruttare la classe ColumnTransformer,
# questa richiede in input una lista di tuple, in cui ogni tupla corrisponde ad una trasformazione da eseguire su una
# colonna che contiene i seguenti elementi:
#
#   Una stringa che indica un nome arbitrario da dare alla trasformazione
#   Il trasformatore istanziato, cioè l'oggetto che vogliamo usare per eseguire la trasformazione.
#   Una lista con gli indici delle colonne alla quale vogliamo applicare la trasformazione.
#
# Il comportamente sulle colonne non trasformate viene definito tramite il parametro remainder,
# se questo è uguale a "drop" esse verranno rimosse, se invece è uguale a "passthrough"
# verranno passate immutate all'array numpy di output del ColumnTransformer.
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

X = shirt.values # Otteniamo l'array numpy corrispondente al DataFrame

ct = ColumnTransformer([('colore', OneHotEncoder(), [1])], remainder='passthrough')
X = ct.fit_transform(X)

print(X[:5])
#le prime tre colonne rappresentano la classe-colore (bianco, rosso e verde),
# la quarta la taglia e la quinta il prezzo




