import pandas as pd #importing required packages #data manipulation
import numpy as np#mathematical calculation
from sklearn.decomposition import PCA#to pca calculation
import matplotlib.pyplot as plt#data visualization
from sklearn.preprocessing import scale #normalization
import seaborn#advanced data visualization
from scipy.cluster.hierarchy import linkage#linkage for clustering
import scipy.cluster.hierarchy as sch#hierarchical clustering
from sklearn.cluster import KMeans#kmeans clustering

wine_data = pd.read_csv("C:/Users/hp/Desktop/pca assi/wine.csv")#loading data
new_wine_data = wine_data.drop(["Type"] , axis=1)#droping column Type

new_wine_data.isna().sum()#checking for na values
new_wine_data.isnull().sum()#checking for null values

dups = new_wine_data.duplicated()#checking duplicate data
sum(dups)
new_wine_data = new_wine_data.drop_duplicates()#droping duplicates

new_wine_data_norm = scale(new_wine_data)#scaling the data

pca = PCA(n_components=6)#calculating pca value
new_wine_data_pca = pca.fit_transform(new_wine_data_norm)

var = pca.explained_variance_ratio_#getting variance of pca data

pca.components_#view pca values

cumsum_var = np.cumsum(np.round(var, decimals = 4) * 100)#getiing cummulative variance in percentages

plt.plot(cumsum_var, color = "red")#ploting pca variance 

new_wine_data_pca = pd.DataFrame(new_wine_data_pca)
new_wine_data_pca.columns = "comp0", "comp1", "comp2", "comp3", "comp4", "comp5"#naming pca columns

new_wine_data_pca_final = pd.concat([wine_data.Type, new_wine_data_pca.iloc[:, 0:3]], axis = 1)


seaborn.boxplot(new_wine_data_pca_final.comp0);plt.title("Boxplot");plt.show()#boxploting pca data to check for outliers
seaborn.boxplot(new_wine_data_pca_final.comp1);plt.title("Boxplot");plt.show()
seaborn.boxplot(new_wine_data_pca_final.comp2);plt.title("Boxplot");plt.show()

#removing outliers by replacing limits
IQR = new_wine_data_pca_final["comp2"].quantile(0.75) - new_wine_data_pca_final["comp2"].quantile(0.25)
L_limit_comp2 = new_wine_data_pca_final["comp2"].quantile(0.25) - (IQR * 1.5)
H_limit_comp2 = new_wine_data_pca_final["comp2"].quantile(0.75) + (IQR * 1.5)
new_wine_data_pca_final["comp2"] = pd.DataFrame(np.where(new_wine_data_pca_final["comp2"] > H_limit_comp2 , H_limit_comp2 ,
                                    np.where(new_wine_data_pca_final["comp2"] < L_limit_comp2 , L_limit_comp2 , new_wine_data_pca_final["comp2"])))
seaborn.boxplot(new_wine_data_pca_final.comp2);plt.title('Boxplot');plt.show()

#hierarchical clustering
linkage_complete = linkage(new_wine_data_pca , method="complete" ,metric="euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram for complete linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(linkage_complete , leaf_rotation= 0 , leaf_font_size= 10)
plt.show()

#kmeans clustering
#getting cluster values for the cluster from 2 - 9
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(new_wine_data_pca_final)
    TWSS.append(kmeans.inertia_)
    
TWSS
#elbow curve or screeplot for choosing better cluster value
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

#kmeans model building
model_new_wine_data_pca_final = KMeans(n_clusters = 3)
model_new_wine_data_pca_final.fit(new_wine_data_pca_final)

#storing cluster value in original dataframe
model_new_wine_data_pca_final.labels_ # getting the labels of clusters assigned to each row 
cluster_new_wine_data_pca_final = pd.Series(model_new_wine_data_pca_final.labels_)  # converting numpy array into pandas series object 
new_wine_data_pca_final['cluster'] = cluster_new_wine_data_pca_final

#replacing index of column
new_wine_data_pca_final = new_wine_data_pca_final.iloc[: , [4 , 0 , 1 , 2 , 3]]

#renaming original cluster data to match with calculated cluster data
#here the given clustered data is almost similar to the calcluated cluster data found out using pca values
new_wine_data_pca_final['Type'].replace({1:0 , 2:2 , 3:1} , inplace=True)

#####################################Problem2################################################
#importing required packages
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
import seaborn
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans

#loading the dataframe
heart_data = pd.read_csv("C:/Users/hp/Desktop/pca assi/heart disease.csv")
new_heart_data = heart_data.drop(["target"] , axis=1)

#checking for na and null values
new_heart_data.isna().sum()
new_heart_data.isnull().sum()

#checking for duplicates and removing duplicates if any
dups = new_heart_data.duplicated()
sum(dups)
new_heart_data = new_heart_data.drop_duplicates()

#scaling the data
new_heart_data_norm = scale(new_heart_data)

#caluclating pca values
pca = PCA(n_components=6)
new_heart_data_pca = pca.fit_transform(new_heart_data_norm)

#caluclatng variance for the pca values
var = pca.explained_variance_ratio_

#viewing pca values
pca.components_

#caluclating cummulative varinace of the pca values in percentagess
cumsum_var = np.cumsum(np.round(var, decimals = 4) * 100)

#ploting the variance data
plt.plot(cumsum_var, color = "red")

new_heart_data_pca = pd.DataFrame(new_heart_data_pca)
new_heart_data_pca.columns = "comp0", "comp1", "comp2", "comp3", "comp4", "comp5"

#taking only first 3 pca components for further calulation as metioned in the question
new_heart_data_pca_final = new_heart_data_pca.iloc[:, 0:3]

#ploting boxplot to check for the outliers
seaborn.boxplot(new_heart_data_pca_final.comp0);plt.title("Boxplot");plt.show()
seaborn.boxplot(new_heart_data_pca_final.comp1);plt.title("Boxplot");plt.show()
seaborn.boxplot(new_heart_data_pca_final.comp2);plt.title("Boxplot");plt.show()

#removing outliers
IQR = new_heart_data_pca_final["comp1"].quantile(0.75) - new_heart_data_pca_final["comp1"].quantile(0.25)
L_limit_comp1 = new_heart_data_pca_final["comp1"].quantile(0.25) - (IQR * 1.5)
H_limit_comp1 = new_heart_data_pca_final["comp1"].quantile(0.75) + (IQR * 1.5)
new_heart_data_pca_final["comp1"] = pd.DataFrame(np.where(new_heart_data_pca_final["comp1"] > H_limit_comp1 , H_limit_comp1 ,
                                    np.where(new_heart_data_pca_final["comp1"] < L_limit_comp1 , L_limit_comp1 , new_heart_data_pca_final["comp1"])))
seaborn.boxplot(new_heart_data_pca_final.comp1);plt.title('Boxplot');plt.show()

IQR = new_heart_data_pca_final["comp2"].quantile(0.75) - new_heart_data_pca_final["comp2"].quantile(0.25)
L_limit_comp2 = new_heart_data_pca_final["comp2"].quantile(0.25) - (IQR * 1.5)
H_limit_comp2 = new_heart_data_pca_final["comp2"].quantile(0.75) + (IQR * 1.5)
new_heart_data_pca_final["comp2"] = pd.DataFrame(np.where(new_heart_data_pca_final["comp2"] > H_limit_comp2 , H_limit_comp2 ,
                                    np.where(new_heart_data_pca_final["comp2"] < L_limit_comp2 , L_limit_comp2 , new_heart_data_pca_final["comp2"])))
seaborn.boxplot(new_heart_data_pca_final.comp2);plt.title('Boxplot');plt.show()

#linkage for the hierarchucal clustering and ploting dendrogram
linkage_complete = linkage(new_heart_data_pca_final, method="complete" ,metric="euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram for complete linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(linkage_complete , leaf_rotation= 0 , leaf_font_size= 10)
plt.show()

#Kmeans model building
#getting cluster values for the cluster from 2 - 9
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(new_heart_data_pca_final)
    TWSS.append(kmeans.inertia_)
    
TWSS

#elbowcurve ploting to choose best cluster value
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

#model buildng
model_new_heart_data_pca_final = KMeans(n_clusters = 3)
model_new_heart_data_pca_final.fit(new_heart_data_pca_final)
model_new_heart_data_pca_final.labels_ # getting the labels of clusters assigned to each row 
cluster_new_heart_data_pca_final = pd.Series(model_new_heart_data_pca_final.labels_)  # converting numpy array into pandas series object 
new_heart_data_pca_final['cluster'] = cluster_new_heart_data_pca_final
new_heart_data_pca_final = new_heart_data_pca_final.iloc[: , [3 , 0 , 1 , 2]]

#since the original cluster is given in only 2 clusters([target]) in question, 2 cluster KMeans is calcuated

model_new_heart_data_pca_final = KMeans(n_clusters = 2)
model_new_heart_data_pca_final.fit(new_heart_data_pca_final)
model_new_heart_data_pca_final.labels_ # getting the labels of clusters assigned to each row 
cluster_new_heart_data_pca_final = pd.Series(model_new_heart_data_pca_final.labels_)  # converting numpy array into pandas series object 
new_heart_data_pca_final['cluster'] = cluster_new_heart_data_pca_final
new_heart_data_pca_final = new_heart_data_pca_final.iloc[: , [3 , 0 , 1 , 2]]
new_heart_data_pca_final1 = pd.concat([heart_data.target, new_heart_data_pca_final.iloc[:, 0:4]], axis = 1)
new_heart_data_pca_final1['target'].replace({1:0 , 0:1} , inplace=True)
#comparison is done and found out that the given cluster[target] in question is different to cluster values for pca components