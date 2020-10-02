##############################  Analyzing HOUSE PRICES DATA

##############################  LIBRARIES  AND IMPORT DATA ##################################

library(tidyverse)
library(caret)
library(skimr)
library(xgboost)
library(e1071)

cat("\014")
rm(list = ls())

source("fillingMethodJulien.R")
source("dealingWithNA.R")
source("./ausiliaryFunctions/convertToFactorDataset.R")
source("./ausiliaryFunctions/scaleNumericalData.R")
source("./ausiliaryFunctions/percentageNumCol.R")
source("./ausiliaryFunctions/solveSkewness.R")
source("./ausiliaryFunctions/removeNumericalCorrelation.R")
source("./ausiliaryFunctions/understandingNA.R")

trainset<-read.csv2("./input/train.csv", sep =",",stringsAsFactors = FALSE)
testset<-read.csv2("./input/test.csv", sep =",",stringsAsFactors = FALSE)

# Since test dataset has no “Saleprice” variable. We will create it and then combine.
testset$SalePrice <- rep(NA,nrow(testset))
dataset<-bind_rows(trainset,testset)
dataset<-dataset %>% select(-Id)
SalePrice<-trainset$SalePrice

# trainset will not have the target variable!
dataset<-dataset %>% select(-SalePrice)

# this is numerical but it's a chategorical
dataset$MSSubClass<-dataset$MSSubClass %>% as.character

namesFactor<- names(dataset)[which(sapply(dataset, is.character))] # to mantain memory after one hot encounding what are the factory variables

# some numerical information
#Count the number of columns that consists of text data
sum(sapply(dataset[,1:79], typeof) == "character")
#Count the number of columns that consists of numerical data
sum(sapply(dataset[,1:79], typeof) == "integer")
percentageNumCol(dataset) # to see how if I will remove to many columns


##############################  DEALING WITH THE NA VALUES ##################################

dataset<-understandingNA(dataset)

for ( i in 1 : ncol(dataset)){
  if(class(dataset[,i])=="character") dataset[,i]<- dataset[,i] %>% as.factor
}

##############################  ONE-HOT ENCODING

# One-Hot Encoding

dummies_modelFullRank <- dummyVars( ~ ., data=dataset, fullRank = TRUE)
data_matFullRank <- predict(dummies_modelFullRank, newdata = dataset)
#dim(data_matFullRank)

method <- "julien"
dataFullRank<-data.frame(data_matFullRank)

dataFullRank<- dealingWithNA(dataFullRank, "julien", ncol(trainset), namesFactor)
saveRDS(dataFullRank,"./dataSaved/dataFullRank.RDS")
#dataFullRank<-readRDS("./dataSaved/dataFullRank.RDS")

# at this point i've the columns with NA filled and the oldest one with near zero variance

##############################  SOLVING PROBLEM OF SKEWNESS ##################################

par(mfrow=c(1,1))
#hist(SalePrice)
responseSkew <- BoxCoxTrans(SalePrice)
SalePriceTrans <- predict(responseSkew, SalePrice)
#hist(SalePriceTrans)

dataFullRankSym <- solveSkewness(dataFullRank)


##############################  CORRELATIONS  ##############################  

# we remove predictors with too big correlations

dataFullRankSymNoCor<-removeNumericalCorrelation(dataFullRankSym)

dummies_model <- dummyVars(~ ., data=dataFullRankSymNoCor)
dataFullRankSymNoCor_mat <- predict(dummies_model, newdata = dataFullRankSymNoCor)

##############################  DEFINING A MACHINE LEARNING MODEL  ##############################  

##############################  CREATING DATA BASE  ##############################  


training<-dataFullRankSymNoCor_mat[1:nrow(trainset),]
trainingIndex<-createDataPartition(SalePrice, p = 0.8, list = FALSE)
# learning 
learning<-training[trainingIndex,]
SalePriceTransLearning<-SalePriceTrans[trainingIndex]
# validation
validation <- training[-trainingIndex,]
SalePriceTransValidation<-SalePriceTrans[-trainingIndex]

# Define the training control
fitControl <- trainControl(
  method = 'cv',                   # k-fold cross validation
  number = 5,                   # number of folds
  allowParallel = TRUE
) 

######## random forest -----

rf <- train(learning, SalePriceTransLearning,
                   method='rf',
                   tuneLength=19, 
                   trControl = fitControl)

saveRDS(rf,"./dataSaved/randomForest/rf.RDS")
rf<-readRDS("./dataSaved/randomForest/rf.RDS")
plot(rf)
rf$bestTune

predictValidation_RF<-predict(rf,validation)
hist(exp(predictValidation_RF)-exp(SalePriceTransValidation))
plot(exp(predictValidation_RF)-exp(SalePriceTransValidation))
abline(h = 0, col = "red")

plot(exp(SalePriceTransValidation),exp(predictValidation_RF))
abline(a=0, b= 1, col = "red")


########  XGBOOST -----

set.seed(123)
# different part of tuningof xgboost : number of trees

hyper_grid <- expand.grid(
  nrounds =  seq(from = 200, to = 1000, by = 50),
  eta = .1,
  gamma = 0,
  max_depth = 5,
  min_child_weight = 1,
  subsample = .8, 
  colsample_bytree = 0.8
)


xgb.fit1 <- train(learning, SalePriceTransLearning,
                 method = "xgbTree",
                 trControl= fitControl,
                 tuneGrid = hyper_grid,
                 verbose = TRUE,
                 nthread= 4
)
plot(xgb.fit1)
saveRDS(xgb.fit1,"./dataSaved/xgboost/xgb.fit1.RDS")
xgg.fit1<-readRDS("./dataSaved/xgboost/xgb.fit1.RDS")

hyper_grid <- expand.grid(
  nrounds =  xgb.fit1$bestTune$nrounds,
  eta = .1,
  gamma = 0,
  max_depth = c(2,3,10),
  min_child_weight = c(1,2,6),
  subsample = .8, 
  colsample_bytree = 0.8
)

xgb.fit2 <- train(learning, SalePriceTransLearning,
                  method = "xgbTree",
                  trControl= fitControl,
                  tuneGrid = hyper_grid,
                  verbose = TRUE,
                  nthread= 4
)
plot(xgb.fit2)
saveRDS(xgb.fit2,"./dataSaved/xgboost/xgb.fit2.RDS")
xgg.fit2<-readRDS("./dataSaved/xgboost/xgb.fit2.RDS")

hyper_grid <- expand.grid(
  nrounds =  xgb.fit1$bestTune$nrounds,
  eta = .1,
  gamma = seq(1/10,1,1/10),
  max_depth = xgb.fit2$bestTune$max_depth,
  min_child_weight = xgb.fit2$bestTune$min_child_weight,
  subsample = .8, 
  colsample_bytree = 0.8
)

xgb.fit3 <- train(learning, SalePriceTransLearning,
                  method = "xgbTree",
                  trControl= fitControl,
                  tuneGrid = hyper_grid,
                  verbose = TRUE,
                  nthread= 4
)
plot(xgb.fit3)
saveRDS(xgb.fit3,"./dataSaved/xgboost/xgb.fit3.RDS")
xgg.fit3<-readRDS("./dataSaved/xgboost/xgb.fit3.RDS")

hyper_grid <- expand.grid(
  nrounds =  xgb.fit1$bestTune$nrounds,
  eta = .1,
  gamma = xgb.fit3$bestTune$gamma,
  max_depth = xgb.fit2$bestTune$max_depth,
  min_child_weight = xgb.fit2$bestTune$min_child_weight,
  subsample = seq(6/10,1.,1/10), 
  colsample_bytree = seq(6/10,1.,1/10)
)
xgb.fit4 <- train(learning, SalePriceTransLearning,
                  method = "xgbTree",
                  trControl= fitControl,
                  tuneGrid = hyper_grid,
                  verbose = TRUE,
                  nthread= 4
)
plot(xgb.fit4)
saveRDS(xgb.fit4,"./dataSaved/xgboost/xgb.fit4.RDS")
xgb.fit4<-readRDS("./dataSaved/xgboost/xgb.fit4.RDS")

hyper_grid <- expand.grid(
  nrounds =  5000,
  eta = 0.01,
  gamma = xgb.fit3$bestTune$gamma,
  max_depth = xgb.fit2$bestTune$max_depth,
  min_child_weight = xgb.fit2$bestTune$min_child_weight,
  subsample = xgb.fit4$bestTune$subsample, 
  colsample_bytree = xgb.fit4$bestTune$colsample_bytree
)
xgb.fit5 <- train(learning, SalePriceTransLearning,
                  method = "xgbTree",
                  trControl= fitControl,
                  tuneGrid = hyper_grid,
                  verbose = TRUE,
                  nthread= 4
)
saveRDS(xgb.fit5,"./dataSaved/xgboost/xgb.fit5.RDS")

hyper_grid <- expand.grid(
  nrounds =  seq(from = 200, to = 5000, by = 500),
  eta = c(.01, .05, .1, .3),
  gamma = 0,
  max_depth = c(1, 3, 5, 7),
  min_child_weight = c(1, 3, 5, 7),
  subsample = c(.65, .8, 1), 
  colsample_bytree = c(.8, .9, 1)
)
xgb.fit6 <- train(learning,SalePriceTransLearning,
                 method = "xgbTree",
                 trControl= fitControl,
                 tuneGrid = hyper_grid,
                 verbose = TRUE,
                 nthread= 4
)
saveRDS(xgb.fit6,"./dataSaved/xgboost/xgb.fit6.RDS")

xgb.bestGreedy<-xgb.fit4
xgb.bestTot<-xgb.fit6
xgb.fit4$bestTune
xgb.fit6$bestTune


############# PREDICT WITH XGBOOST -----

testAllColumnsSymNoCorScaled<-dataAllColumns[(nrow(trainset)+1):(nrow(testset)+nrow(trainset)),]
dummies_model <- dummyVars(~ ., data=testAllColumnsSymNoCorScaled)
testing6_mat <- predict(dummies_model, newdata = testAllColumnsSymNoCorScaled)

varimp_xgboost <- varImp(xgbtrMat6)
plot(varimp_xgboost, main="Variable Importance with XGBOOST")
predicted <- predict(xgbtrMat6, testing6_mat)
predictedNormal<-exp(predicted)
prediction<-data.frame(Id = testset %>% select(Id), SalePrice = predictedNormal)
write.csv(prediction,"submission.csv", row.names = FALSE)
