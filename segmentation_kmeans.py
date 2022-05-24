#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale

from matplotlib.path import get_path_collection_extents
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.datasets import make_blobs
from tqdm import tqdm_notebook as tqdm
from scipy.cluster import hierarchy
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin
from sklearn.metrics import silhouette_score


# In[3]:


df = pd.read_excel('/Users/natalyakrauze/Desktop/диплом/final_clean_dataset.xlsx')


# In[ ]:


df_experiement = pd.read_excel('/Users/natalyakrauze/Desktop/google_survey_initial.xlsx')


# In[4]:


d = dict()
i = 0
for el in list(df.columns):
    d[el] = list(df.dtypes)[i]
    i += 1
d


# In[5]:


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


# In[6]:


df = DataFrameImputer().fit_transform(df)


# In[7]:


df['Gender_binary'] = np.where(df['Gender']=='Мужчина', '1', '0')


# In[8]:


df = df.drop(['Gender'], axis=1)


# In[9]:


df = df.join(pd.get_dummies(df.EduLevel))


# In[10]:


df = df.drop(['EduLevel', 'fixed_uni_name_1', 'fixed_uni_name_2'], axis=1)


# In[11]:


data = df.values


# # Normalization

# In[12]:


scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)


# In[ ]:


scaler = StandardScaler()
scaled_data_experiement = scaler.fit_transform(df_experiement)


# In[13]:


list(df.columns)


# In[14]:


df = pd.DataFrame(scaled_data)
df.columns = ['ExpPeriod',
 'Salary',
 'Age',
 'Москва и МО',
 'Санкт-Петербург',
 'Регионы',
 'N_places',
 'N_langs',
 'Ready_4_business_trip',
 'Graduation',
 'N_Unis',
 'полная занятость',
 'проектная работа',
 'частичная занятость',
 'стажировка',
 'удаленная работа',
 'гибкий график',
 'Английский',
 'Русский',
 'Китайский',
 'SQL',
 'Python',
 'Pandas',
 'Разработка технических заданий',
 'MS Excel',
 'MS Word',
 'Ответственность',
 'Аналитический склад ума',
 'аналитик',
 'системный администратор',
 'финансовый аналитик',
 'бизнес аналитик',
 'business analyst',
 'data scientist',
 'аналитик bi',
 'junior',
 'программист',
 'developer',
 'project manager',
 'teamlead',
 'product manager',
 'технический директор',
 'начальник отдела',
 'руководитель проектов',
 'руководитель отдела',
 'it',
 'web',
 'uni_1_code',
 'uni_2_code',
 'Gender_binary',
 'Высшее образование',
 'Высшее образование (Бакалавр)',
 'Высшее образование (Доктор наук)',
 'Высшее образование (Кандидат наук)',
 'Высшее образование (Магистр)',
 'Неоконченное высшее образование',
 'Среднее образование',
 'Среднее специальное образование']


# ## CORELLATION 

# In[86]:


df.corr()


# In[87]:


plt.figure(figsize=(40, 30))
sns.heatmap(df.corr(), annot=True)
plt.show()


# ## Determine number of clusters

# In[16]:


Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df)
    Sum_of_squared_distances.append(km.inertia_)


# In[17]:


plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[18]:


def optimalK(data, nrefs=3, maxClusters=12):
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters)):
       # data: our array 
       # nrefs: number of sample reference datasets to create
       # maxClusters: Maximum number of clusters to test for

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            
            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)
        
        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)

    return (gaps.argmax() + 1, resultsdf) 
    


# In[19]:


k, gapdf = optimalK(df, nrefs=3, maxClusters=8)
print('Optimal k is: ', k)


# In[20]:


plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
plt.grid(True)
plt.xlabel('Cluster Count')
plt.ylabel('Gap Value')
plt.title('Gap Values by Cluster Count')
plt.show()


# In[14]:


pca = PCA(n_components = 2)
X_principal = pca.fit_transform(data)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']


# In[19]:


from sklearn.cluster import AgglomerativeClustering


# In[20]:


ac2 = AgglomerativeClustering(n_clusters = 2)
ac3 = AgglomerativeClustering(n_clusters = 3)
ac4 = AgglomerativeClustering(n_clusters = 4)
ac5 = AgglomerativeClustering(n_clusters = 5)


# In[ ]:


k = [2, 3, 4, 5]
# Appending the silhouette scores of the different models to the list
silhouette_scores = []
silhouette_scores.append(
silhouette_score(X_principal, ac2.fit_predict(X_principal)))
silhouette_scores.append(
silhouette_score(X_principal, ac3.fit_predict(X_principal)))
silhouette_scores.append(
silhouette_score(X_principal, ac4.fit_predict(X_principal)))
silhouette_scores.append(
silhouette_score(X_principal, ac5.fit_predict(X_principal)))
# Plotting a bar graph to compare the results
plt.bar(k, silhouette_scores)
plt.xlabel('Number of clusters', fontsize = 20)
plt.ylabel('S(i)', fontsize = 20)
plt.show()


# ## Segmentation: k-means 5 clusters

# In[21]:


k = 5
kmeans = KMeans(n_clusters=k).fit(df)


# Experiement

# In[22]:


pca = PCA(n_components = 2)
pca.fit(df)


# In[ ]:


k = 5
kmeans = KMeans(n_clusters=k).fit(scaled_data_experiement)


# In[ ]:


pca = PCA(n_components = 2)
pca.fit(scaled_data_experiement)


# In[23]:


data1 = pca.transform(df)
print(data1.shape)


# In[ ]:


data2 = pca.transform(scaled_data_experiement)
print(data2.shape)


# In[26]:


colors = np.random.randint(0, 255, size=(k,4))/255
c_arr = np.array(list(map(lambda x: colors[x], list(kmeans.labels_))))


# In[27]:


plt.scatter(data1[:, 0], data1[:, 1], c=c_arr)


# In[30]:


cluster_1 = []
cluster_2 = []
cluster_3 = []
cluster_4 = []
cluster_5 = []

for i in range(data.shape[0]):
    if kmeans.labels_[i] == 0:
        cluster_1.append(data[i])
    elif kmeans.labels_[i] == 1:
        cluster_2.append(data[i])
    elif kmeans.labels_[i] == 2:
        cluster_3.append(data[i])
    elif kmeans.labels_[i] == 3:
        cluster_4.append(data[i])
    elif kmeans.labels_[i] == 4:
        cluster_5.append(data[i])


# In[ ]:


cluster_1 = []
cluster_2 = []
cluster_3 = []
cluster_4 = []
cluster_5 = []

for i in range(data.shape[0]):
    if kmeans.labels_[i] == 0:
        cluster_1.append(data2[i])
    elif kmeans.labels_[i] == 1:
        cluster_2.append(data2[i])
    elif kmeans.labels_[i] == 2:
        cluster_3.append(data2[i])
    elif kmeans.labels_[i] == 3:
        cluster_4.append(data2[i])
    elif kmeans.labels_[i] == 4:
        cluster_5.append(data2[i])


# In[31]:


df = pd.DataFrame(df)


# In[33]:


data = pd.DataFrame(data)
data


# In[34]:


data.columns = ['ExpPeriod',
 'Salary',
 'Age',
 'Москва и МО',
 'Санкт-Петербург',
 'Регионы',
 'N_places',
 'N_langs',
 'Ready_4_business_trip',
 'Graduation',
 'N_Unis',
 'полная занятость',
 'проектная работа',
 'частичная занятость',
 'стажировка',
 'удаленная работа',
 'гибкий график',
 'Английский',
 'Русский',
 'Китайский',
 'SQL',
 'Python',
 'Pandas',
 'Разработка технических заданий',
 'MS Excel',
 'MS Word',
 'Ответственность',
 'Аналитический склад ума',
 'аналитик',
 'системный администратор',
 'финансовый аналитик',
 'бизнес аналитик',
 'business analyst',
 'data scientist',
 'аналитик bi',
 'junior',
 'программист',
 'developer',
 'project manager',
 'teamlead',
 'product manager',
 'технический директор',
 'начальник отдела',
 'руководитель проектов',
 'руководитель отдела',
 'it',
 'web',
 'uni_1_code',
 'uni_2_code',
 'Gender_binary',
 'Высшее образование',
 'Высшее образование (Бакалавр)',
 'Высшее образование (Доктор наук)',
 'Высшее образование (Кандидат наук)',
 'Высшее образование (Магистр)',
 'Неоконченное высшее образование',
 'Среднее образование',
 'Среднее специальное образование']


# In[36]:


labels = pd.DataFrame(kmeans.labels_)


# In[37]:


data['Кластер'] = labels


# In[42]:


data.to_excel('kmeans_5.xlsx')


# In[43]:


means = data.groupby('Кластер').mean()


# In[45]:


means.to_csv('kmeans_5_pivot.csv', index=False, sep=';', encoding='utf-8-sig')


# # Silhouette score

# In[ ]:


score = silhouette_score(X, km.labels_, metric='euclidean')
print('Silhouetter Score: %.3f' % score)


# ## Segmentation: k-means 7 clusters

# In[61]:


k = 7
kmeans = KMeans(n_clusters=k).fit(df)


# In[62]:


pca = PCA(n_components = 2); pca.fit(df)


# In[63]:


data1 = pca.transform(df)
print(data1.shape)


# In[64]:


colors = np.random.randint(0, 255, size=(k,4))/255
c_arr = np.array(list(map(lambda x: colors[x], list(kmeans.labels_))))


# In[65]:


plt.scatter(data1[:, 0], data1[:, 1], c=c_arr)


# In[66]:


cluster_1 = []
cluster_2 = []
cluster_3 = []
cluster_4 = []
cluster_5 = []
cluster_6 = []
cluster_7 = []

for i in range(data.shape[0]):
    if kmeans.labels_[i] == 0:
        cluster_1.append(data[i])
    elif kmeans.labels_[i] == 1:
        cluster_2.append(data[i])
    elif kmeans.labels_[i] == 2:
        cluster_3.append(data[i])
    elif kmeans.labels_[i] == 3:
        cluster_4.append(data[i])
    elif kmeans.labels_[i] == 4:
        cluster_5.append(data[i])
    elif kmeans.labels_[i] ==5:
        cluster_6.append(data[i])
    elif kmeans.labels_[i] == 6:
        cluster_7.append(data[i])


# In[67]:


df = pd.DataFrame(df)
data = pd.DataFrame(data)


# In[68]:


data.columns = ['ExpPeriod',
 'Salary',
 'Age',
 'Москва и МО',
 'Санкт-Петербург',
 'Регионы',
 'N_places',
 'N_langs',
 'Ready_4_business_trip',
 'Graduation',
 'N_Unis',
 'полная занятость',
 'проектная работа',
 'частичная занятость',
 'стажировка',
 'удаленная работа',
 'гибкий график',
 'Английский',
 'Русский',
 'Китайский',
 'SQL',
 'Python',
 'Pandas',
 'Разработка технических заданий',
 'MS Excel',
 'MS Word',
 'Ответственность',
 'Аналитический склад ума',
 'аналитик',
 'системный администратор',
 'финансовый аналитик',
 'бизнес аналитик',
 'business analyst',
 'data scientist',
 'аналитик bi',
 'junior',
 'программист',
 'developer',
 'project manager',
 'teamlead',
 'product manager',
 'технический директор',
 'начальник отдела',
 'руководитель проектов',
 'руководитель отдела',
 'it',
 'web',
 'uni_1_code',
 'uni_2_code',
 'Gender_binary',
 'Высшее образование',
 'Высшее образование (Бакалавр)',
 'Высшее образование (Доктор наук)',
 'Высшее образование (Кандидат наук)',
 'Высшее образование (Магистр)',
 'Неоконченное высшее образование',
 'Среднее образование',
 'Среднее специальное образование']


# In[69]:


labels = pd.DataFrame(kmeans.labels_)


# In[70]:


data['Кластер'] = labels


# In[71]:


data.to_excel('kmeans_7.xlsx')


# In[72]:


means = data.groupby('Кластер').mean()
means.to_csv('kmeans_7_pivot.csv', index=False, sep=';', encoding='utf-8-sig')


# ## Hierrarhical

# In[ ]:


import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms")
dend = shc.dendrogram(shc.linkage(df, method='ward'))


# In[ ]:


from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)


# In[ ]:


plt.figure(figsize=(10, 7))
plt.scatter(df[:,0], df[:,1], c=cluster.labels_, cmap='rainbow')


# In[ ]:




