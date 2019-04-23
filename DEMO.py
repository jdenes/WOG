import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import json

data = pd.read_csv("1-rne-cm.txt", sep='\t', encoding='ISO-8859-1')

data.shape

data.head()

data['Libellé de département (Maires)'].value_counts().plot(kind='bar', figsize=(20,5))

data['Code sexe'].value_counts(normalize=True)

data['Libellé de la profession'].value_counts().head(30).plot(kind='barh', figsize=(20,8))

data["Nationalité de l'élu"].value_counts(normalize=True)

data['Date de naissance'].describe()

data['Date de naissance clean'] = data['Date de naissance'].astype("datetime64")

data['Date de naissance clean'].describe()

data.groupby(data['Date de naissance clean'].dt.year).count()['Code sexe'].plot(figsize=(15, 5))

data.groupby(['Libellé de département (Maires)', 'Code sexe']).size().unstack().plot.bar(figsize=(25, 5))
