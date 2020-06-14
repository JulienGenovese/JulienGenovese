###### Analyzing Mushroom data


##############################  LIBRARIES  AND IMPORT DATA ##################################
library(tidyverse)
library(randomForest)
library(forecast)
library(tree)
library(rsample)      # data splitting 

library(corrplot)
library(RColorBrewer)
library(vcd)

cat("\014")
rm(list = ls())

source("ausiliaryFunctions.R")

pathinput<-"mushrooms.csv"
dataset<-read.csv2(pathinput , sep =",",stringsAsFactors = FALSE) 
dataset<-data.frame(apply(dataset,2,FUN = as.factor))

set.seed(123)
mr_split <- initial_split(dataset, prop = .7)
mr_train <- training(mr_split)
mr_test  <- testing(mr_split)

##############################  GOING INTO INTO DATASET ##################################

head(mr_train)
str(mr_train)
mr_train <- mr_train %>% select(-"veil.type") # we have only a type in this dataset

colSums(is.na(mr_train)) 
if (any(mr_train=='?')) print('At least one null value is in df_train')
df_train<- mr_train %>% select(-"stalk.root")

##############################  FEATURE ENGINEERING  ##################################
# we want to create new columns but we don't know how.
# random combinations
k<-ncol(df_train)+1
size<-ncol(df_train)
for(i in 2: size){
  if(i<size){
    for(j in (i+1):size){
      df_train[,k]<-combine_column(df_train[,i],df_train[,j])
      colnames(df_train)[k]<-paste(colnames(df_train)[i],colnames(df_train)[j])
      k<-k+1
    }
  }
}
df_train<-data.frame(apply(df_train,2,FUN = as.factor))

# Correlation matrix
cor_matrix <- matrix(ncol = ncol(df_train),
                  nrow = ncol(df_train),
                  dimnames = list(names(df_train), 
                                  names(df_train)))

cor_matrix <- calculate_cramer(cor_matrix,df_train) # correlation within chategorical variables
cor_matrix2<-cor_matrix[1:23,1:23]
corrplot(cor_matrix2, type="upper", order="original",tl.cex = 0.8,
         col=brewer.pal(n=8, name="RdYlBu"))

corClass<-cor_matrix[1,1:ncol(cor_matrix)]
plot(corClass)
threshold<-0.6 # we take only the features with a strong correlation

importantVariables<-colnames(df_train)[corClass>threshold] 
df_addFeatures<-df_train %>% select(importantVariables)
d<-rep(NA,1,ncol(df_addFeatures))
remove<-c()
for(i in 1:ncol(df_addFeatures)) {
  d[i]<-(length(levels(df_addFeatures[,i])))
  if(d[i]>20){
    remove<-c(remove,i)
  }
}
# we remove the features with too many chategorical variables
plot(d)
df_addFeatures<-df_addFeatures[,-remove]
columnsnames<-colnames(df_addFeatures)

##############################  PREDICTION PART ##################################
# in this part we create again the two dataset because we need the same levels for both test and 
# train.

totdataset<-bind_rows(mr_train,mr_test) 

totdataset <- totdataset %>% select(-c("veil.type","stalk.root")) # we have only a type in this dataset
k<-ncol(totdataset)+1
size<-ncol(totdataset)
for(i in 2: size){
  if(i<size){
    for(j in (i+1):size){
      totdataset[,k]<-combine_column(totdataset[,i],totdataset[,j])
      colnames(totdataset)[k]<-paste(colnames(totdataset)[i],colnames(totdataset)[j])
      k<-k+1
    }
  }
}
totdataset<-data.frame(apply(totdataset,2,FUN = as.factor))
totdataset<-totdataset %>% select(columnsnames)

trainset<-totdataset[1:nrow(df_addFeatures),]
testset<-totdataset[(nrow(df_addFeatures)+1):nrow(totdataset),]
####### Here we train the model 
rf<-tree(class~., data = trainset)
# some summaries of the model
rf
plot(rf)
text(rf,pretty = 0)
summary(rf)

# we see that the prediciton is easy. We want to see if, from histograms we can see the reason
barplot(prop.table(table(df_addFeatures$odor.stalk.color.below.ring[df_addFeatures$class=="e"])))
barplot(prop.table(table(df_addFeatures$odor.stalk.color.below.ring[df_addFeatures$class=="p"])))

##############################  ACCURACY PART ##################################

x<-testset %>% select(-"class")
y<-testset$class
predictions<-predict(rf, x, type ="class")
mean(y==predictions)
