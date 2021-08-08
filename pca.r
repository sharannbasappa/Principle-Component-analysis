###################################Problem 1#############################################
install.packages("readr")
library(readr)

wine_data <- read.csv("C:/Users/hp/Desktop/pca assi/wine.csv")
new_wine_data <- wine_data[-1]

sum(is.na(new_wine_data))
sum(is.null(new_wine_data))

dupl<- duplicated(new_wine_data)
sum(dupl)
new_wine_data <- new_wine_data[!dupl , ]

norm_wine_data <- scale(new_wine_data)
norm_wine_data <- data.frame(norm_wine_data)

pca_wine_data <- princomp(norm_wine_data, cor = TRUE, scores = TRUE, covmat = NULL)

loadings(pca_wine_data)

plot(pca_wine_data) # graph showing importance of principal components 

biplot(pca_wine_data)

plot(cumsum(pca_wine_data$sdev * pca_wine_data$sdev) * 100 / (sum(pca_wine_data$sdev * pca_wine_data$sdev)), type = "b")

pca_wine_data$scores
pca_wine_data$scores[, 1:3]

# Top 3 pca scores 
pca_wine_data_final <- as.data.frame(pca_wine_data$scores[, 1:3])

boxplot(pca_wine_data_final$Comp.1)
boxplot(pca_wine_data_final$Comp.2)
boxplot(pca_wine_data_final$Comp.3)

qunt_Comp_3 <- quantile(pca_wine_data_final$Comp.3 , probs = c(.25 , .75))
winso_Comp_3 <- quantile(pca_wine_data_final$Comp.3 , probs = c(.05 , .95) , na.rm = TRUE)
H_Comp_3 <- 1.5*IQR(pca_wine_data_final$Comp.3 , na.rm = TRUE)
pca_wine_data_final$Comp.3[pca_wine_data_final$Comp.3<(qunt_Comp_3[1]-H_Comp_3)] <- winso_Comp_3[1]
pca_wine_data_final$Comp.3[pca_wine_data_final$Comp.3>(qunt_Comp_3[2]+H_Comp_3)] <- winso_Comp_3[2]
boxplot(pca_wine_data_final$Comp.3)

# Scatter diagram
plot(pca_wine_data_final$Comp.1, pca_wine_data_final$Comp.2)
plot(pca_wine_data_final$Comp.2, pca_wine_data_final$Comp.3)
plot(pca_wine_data_final$Comp.1, pca_wine_data_final$Comp.3)

#Model Building hierarchi

Edist <- dist(pca_wine_data_final, method = "euclidean")
Hcl1 <- hclust(Edist , method = "complete")
plot(Hcl1 , hang= -1)
clustered <- cutree(Hcl1,k=3)
rect.hclust(Hcl1, k = 3, border = "red")
clustered <- as.matrix(clustered) 
pca_wine_data_final2 <- data.frame(clustered , pca_wine_data_final)
pca_wine_data_final2 <- data.frame(wine_data$Type,pca_wine_data_final2)
#almost same

#Model Building KMeans
library(readr)

wine_data <- read.csv("C:/Users/hp/Desktop/pca assi/wine.csv")
new_wine_data <- wine_data[-1]

sum(is.na(new_wine_data))
sum(is.null(new_wine_data))

dupl<- duplicated(new_wine_data)
sum(dupl)
new_wine_data <- new_wine_data[!dupl , ]

norm_wine_data <- scale(new_wine_data)
norm_wine_data <- data.frame(norm_wine_data)

pca_wine_data <- princomp(norm_wine_data, cor = TRUE, scores = TRUE, covmat = NULL)

loadings(pca_wine_data)

plot(pca_wine_data) # graph showing importance of principal components 

biplot(pca_wine_data)

plot(cumsum(pca_wine_data$sdev * pca_wine_data$sdev) * 100 / (sum(pca_wine_data$sdev * pca_wine_data$sdev)), type = "b")

pca_wine_data$scores
pca_wine_data$scores[, 1:3]

# Top 3 pca scores 
pca_wine_data_final <- as.data.frame(pca_wine_data$scores[, 1:3])

boxplot(pca_wine_data_final$Comp.1)
boxplot(pca_wine_data_final$Comp.2)
boxplot(pca_wine_data_final$Comp.3)

qunt_Comp_3 <- quantile(pca_wine_data_final$Comp.3 , probs = c(.25 , .75))
winso_Comp_3 <- quantile(pca_wine_data_final$Comp.3 , probs = c(.05 , .95) , na.rm = TRUE)
H_Comp_3 <- 1.5*IQR(pca_wine_data_final$Comp.3 , na.rm = TRUE)
pca_wine_data_final$Comp.3[pca_wine_data_final$Comp.3<(qunt_Comp_3[1]-H_Comp_3)] <- winso_Comp_3[1]
pca_wine_data_final$Comp.3[pca_wine_data_final$Comp.3>(qunt_Comp_3[2]+H_Comp_3)] <- winso_Comp_3[2]
boxplot(pca_wine_data_final$Comp.3)

# Elbow curve to decide the k value
twss <- NULL
for (i in 2:8) {
  twss <- c(twss, kmeans(pca_wine_data_final, centers = i)$tot.withinss)
}
twss

# Look for an "elbow" in the scree plot
plot(2:8, twss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
title(sub = "K-Means Clustering Scree-Plot")

# 3 Cluster Solution
fit_wine_data <- kmeans(pca_wine_data_final, 3) 
str(fit_wine_data)
fit_wine_data$cluster
final_wine_data <- data.frame(fit_wine_data$cluster,wine_data) # Append cluster membership

write_csv(final_wine_data, "Kmeans_wine_data.csv")
getwd()

#############################Problem 2########################################

install.packages("readr")
library(readr)

heart_data <- read.csv("C:/Users/hp/Desktop/pca assi/heart disease.csv")

sum(is.na(heart_data))
sum(is.null(heart_data))

dupl<- duplicated(heart_data)
sum(dupl)
heart_data <- heart_data[!dupl , ]

new_heart_data <- heart_data[-14]

norm_heart_data <- scale(new_heart_data)
norm_heart_data <- data.frame(norm_heart_data)

pca_heart_data <- princomp(norm_heart_data, cor = TRUE, scores = TRUE, covmat = NULL)

loadings(pca_heart_data)

plot(pca_heart_data) # graph showing importance of principal components 

biplot(pca_heart_data)

plot(cumsum(pca_heart_data$sdev * pca_heart_data$sdev) * 100 / (sum(pca_heart_data$sdev * pca_heart_data$sdev)), type = "b")

pca_heart_data$scores
pca_heart_data$scores[, 1:3]

# Top 3 pca scores 
pca_heart_data_final <- as.data.frame(pca_heart_data$scores[, 1:3])

boxplot(pca_heart_data_final$Comp.1)
boxplot(pca_heart_data_final$Comp.2)
boxplot(pca_heart_data_final$Comp.3)

qunt_Comp_2 <- quantile(pca_heart_data_final$Comp.2 , probs = c(.25 , .75))
winso_Comp_2 <- quantile(pca_heart_data_final$Comp.2 , probs = c(.01 , .99) , na.rm = TRUE)
H_Comp_2 <- 1.5*IQR(pca_heart_data_final$Comp.2 , na.rm = TRUE)
pca_heart_data_final$Comp.2[pca_heart_data_final$Comp.2<(qunt_Comp_2[1]-H_Comp_2)] <- winso_Comp_2[1]
pca_heart_data_final$Comp.2[pca_heart_data_final$Comp.2>(qunt_Comp_2[2]+H_Comp_2)] <- winso_Comp_2[2]
boxplot(pca_heart_data_final$Comp.2)

qunt_Comp_3 <- quantile(pca_heart_data_final$Comp.3 , probs = c(.25 , .75))
winso_Comp_3 <- quantile(pca_heart_data_final$Comp.3 , probs = c(.01 , .99) , na.rm = TRUE)
H_Comp_3 <- 1.5*IQR(pca_heart_data_final$Comp.3 , na.rm = TRUE)
pca_heart_data_final$Comp.3[pca_heart_data_final$Comp.3<(qunt_Comp_3[1]-H_Comp_3)] <- winso_Comp_3[1]
pca_heart_data_final$Comp.3[pca_heart_data_final$Comp.3>(qunt_Comp_3[2]+H_Comp_3)] <- winso_Comp_3[2]
boxplot(pca_heart_data_final$Comp.3)

# Scatter diagram
plot(pca_heart_data_final$Comp.1, pca_heart_data_final$Comp.2)
plot(pca_heart_data_final$Comp.2, pca_heart_data_final$Comp.3)
plot(pca_heart_data_final$Comp.1, pca_heart_data_final$Comp.3)

#Model Building hierarchi

Edist <- dist(pca_heart_data_final, method = "euclidean")
Hcl1 <- hclust(Edist , method = "complete")
plot(Hcl1 , hang= -1)
clustered <- cutree(Hcl1,k=3)
rect.hclust(Hcl1, k = 3, border = "red")
clustered <- as.matrix(clustered) 
pca_heart_data_final2 <- data.frame(clustered , pca_heart_data_final)
pca_heart_data_final2 <- data.frame(heart_data$target,pca_heart_data_final2)

#not same
install.packages("readr")
library(readr)

heart_data <- read.csv("C:/Users/hp/Desktop/pca assi/heart disease.csv")

sum(is.na(heart_data))
sum(is.null(heart_data))

dupl<- duplicated(heart_data)
sum(dupl)
heart_data <- heart_data[!dupl , ]

new_heart_data <- heart_data[-14]

norm_heart_data <- scale(new_heart_data)
norm_heart_data <- data.frame(norm_heart_data)

pca_heart_data <- princomp(norm_heart_data, cor = TRUE, scores = TRUE, covmat = NULL)

loadings(pca_heart_data)

plot(pca_heart_data) # graph showing importance of principal components 

biplot(pca_heart_data)

plot(cumsum(pca_heart_data$sdev * pca_heart_data$sdev) * 100 / (sum(pca_heart_data$sdev * pca_heart_data$sdev)), type = "b")

pca_heart_data$scores
pca_heart_data$scores[, 1:3]

# Top 3 pca scores 
pca_heart_data_final <- as.data.frame(pca_heart_data$scores[, 1:3])

boxplot(pca_heart_data_final$Comp.1)
boxplot(pca_heart_data_final$Comp.2)
boxplot(pca_heart_data_final$Comp.3)

qunt_Comp_2 <- quantile(pca_heart_data_final$Comp.2 , probs = c(.25 , .75))
winso_Comp_2 <- quantile(pca_heart_data_final$Comp.2 , probs = c(.01 , .99) , na.rm = TRUE)
H_Comp_2 <- 1.5*IQR(pca_heart_data_final$Comp.2 , na.rm = TRUE)
pca_heart_data_final$Comp.2[pca_heart_data_final$Comp.2<(qunt_Comp_2[1]-H_Comp_2)] <- winso_Comp_2[1]
pca_heart_data_final$Comp.2[pca_heart_data_final$Comp.2>(qunt_Comp_2[2]+H_Comp_2)] <- winso_Comp_2[2]
boxplot(pca_heart_data_final$Comp.2)

qunt_Comp_3 <- quantile(pca_heart_data_final$Comp.3 , probs = c(.25 , .75))
winso_Comp_3 <- quantile(pca_heart_data_final$Comp.3 , probs = c(.01 , .99) , na.rm = TRUE)
H_Comp_3 <- 1.5*IQR(pca_heart_data_final$Comp.3 , na.rm = TRUE)
pca_heart_data_final$Comp.3[pca_heart_data_final$Comp.3<(qunt_Comp_3[1]-H_Comp_3)] <- winso_Comp_3[1]
pca_heart_data_final$Comp.3[pca_heart_data_final$Comp.3>(qunt_Comp_3[2]+H_Comp_3)] <- winso_Comp_3[2]
boxplot(pca_heart_data_final$Comp.3)

# Scatter diagram
plot(pca_heart_data_final$Comp.1, pca_heart_data_final$Comp.2)
plot(pca_heart_data_final$Comp.2, pca_heart_data_final$Comp.3)
plot(pca_heart_data_final$Comp.1, pca_heart_data_final$Comp.3)

# Elbow curve to decide the k value
twss <- NULL
for (i in 2:8) {
  twss <- c(twss, kmeans(pca_heart_data_final, centers = i)$tot.withinss)
}
twss

# Look for an "elbow" in the scree plot
plot(2:8, twss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
title(sub = "K-Means Clustering Scree-Plot")

# 3 Cluster Solution
fit_heart_data <- kmeans(pca_heart_data_final, 2) 
str(fit_heart_data)
fit_heart_data$cluster
final_heart_data <- data.frame(fit_heart_data$cluster,heart_data) # Append cluster membership
final_heart_data <- final_heart_data[ , c(15 ,1,2,3,4,5,6,7,8,9,10,11,12,13,14)]
write_csv(final_heart_data, "Kmeans_heart_data.csv")
getwd()

##########################################END###########################################