### idea --> costruire feature da PCA/tsne

# paper TSNE http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf


### opzione 1--> primi autovettori
### opzione 2--> fare clustering e far diventare la colonna non-supervisionata una feature



library(tidyverse)
library(caret)
library(skimr)

cat("\014")
rm(list = ls())

trainset<-read.csv2("./input/train.csv", sep =",")
testset<-read.csv2("./input/test.csv", sep =",")

skimmed <- skim_to_wide(trainset)
skimmed[, c(1:4, 8:9, 13, 15:15)]

nomecolonne<-colnames(trainset)[colSums(is.na(trainset))>0]
naPerc<-colSums(is.na(trainset))/nrow(trainset)

toMantain<-colnames(trainset)[!naPerc>0.5]
trainset <- trainset[toMantain] %>% select(-"Id")

is.categorical <- function(x){
  is.character(x) | is.factor(x)
}
category <- which(sapply(trainset, is.categorical))
tabella<-lapply(
  names(category), function(x){
    print(x)
    print(table(trainset[, x]))
  }
)





categoryToHide = names(testset)[sapply(testset, class) == "factor"]
data <- testset %>% select(-categoryToHide)

#data <-data %>% select(2:4)
for(i in 1:ncol(data)){
  data[is.na(data[,i]), i] <- mean(data[,i], na.rm = TRUE)
}





######################################## PCA ######################################################

scaled_data = scale(data, scale=TRUE)
# Boxplot
x11()
layout(matrix(c(2,2,1,1),2,byrow=T))
boxplot(scaled_data)
boxplot(data)

pca <- prcomp(scaled_data, center=TRUE)
str(pca)
summary(pca)
plot(pca)


x11()
layout(matrix(c(2,3,1,3),2,byrow=T))
barplot((pca)$sdev^2, las=2, main='Componenti principali', ylim=c(0,10), ylab='Variances')
barplot(sapply(data,sd)^2, las=2, main='Variabili originarie', ylab='Variances')
plot(cumsum(pca$sdev^2)/sum(pca$sde^2), type='b', axes=F, xlab='numero di componenti', ylab='contributo alla varianza totale', ylim=c(0,1))
abline(h=1, col='blue')
abline(h=0.8, lty=2, col='blue')
box()
axis(2,at=0:10/10,labels=0:10/10)
axis(1,at=1:ncol(data),labels=1:ncol(data),las=2)

# loadings
pca$rotation # loadings
# I loadings rispecchiano l'osservazione precedente: la prima componente

#Scores
pca$x # scores
#homemade plot degli scores
#plot(pca$x[,1],pca$x[,2])
#install.packages('ggfortify') --> plot tramite ggfortify
library(ggfortify)
autoplot(pca)

x11()
autoplot(pca,loadings = TRUE, loadings.colour = 'blue',
         loadings.label = TRUE, loadings.label.size = 3)


######################################## TSNE ######################################################

library(Rtsne)
tsne = Rtsne(data,dims=2, perplexity=30, theta=0.5) #,pca=TRUE)
dat_tsne = as.data.frame(tsne$Y) 

x11()
#grafico tsne --> vedere con pca = True
ggplot(dat_tsne, aes(x=V1, y=V2)) +  
  geom_point(size=0.25) +
  guides(colour=guide_legend(override.aes=list(size=6))) +
  xlab("") + ylab("") +
  ggtitle("t-SNE") +
  theme_light(base_size=20) +
  theme(axis.text.x=element_blank(),
        axis.text.y=element_blank()) +
  scale_colour_brewer(palette = "Set2")



############################################   clustering  ############################################
#modificare solo il numero di clusters e runnare l'intero codice
N_CLUSTERS = 6

d_tsne_1 = dat_tsne
## keeping original data
d_tsne_1_original=d_tsne_1

## Creating k-means clustering model, and assigning the result to the data used to create the tsne
fit_cluster_kmeans=kmeans(scale(d_tsne_1), N_CLUSTERS)  
d_tsne_1_original$cl_kmeans = factor(fit_cluster_kmeans$cluster)

## Creating hierarchical cluster model, and assigning the result to the data used to create the tsne
fit_cluster_hierarchical=hclust(dist(scale(d_tsne_1)))

## setting 3 clusters as output
d_tsne_1_original$cl_hierarchical = factor(cutree(fit_cluster_hierarchical, k=N_CLUSTERS))  

plot_cluster=function(data, var_cluster, palette)  
{
  ggplot(data, aes_string(x="V1", y="V2", color=var_cluster)) +
    geom_point(size=0.25) +
    guides(colour=guide_legend(override.aes=list(size=6))) +
    xlab("") + ylab("") +
    ggtitle("") +
    theme_light(base_size=20) +
    theme(axis.text.x=element_blank(),
          axis.text.y=element_blank(),
          legend.direction = "horizontal", 
          legend.position = "bottom",
          legend.box = "horizontal") + 
    scale_colour_brewer(palette = palette) 
}

x11()
plot_k=plot_cluster(d_tsne_1_original, "cl_kmeans", "Accent")  
plot_h=plot_cluster(d_tsne_1_original, "cl_hierarchical", "Set1")
## and finally: putting the plots side by side with gridExtra lib...
library(gridExtra)  
grid.arrange(plot_k, plot_h,  ncol=2) 



## Verificare di sotto se vedo una certa similarità anche plottando le prime due componenti principali
x11()
plot(pca$x[,1],pca$x[,2],col = d_tsne_1_original$cl_hierarchical,pch=19)





