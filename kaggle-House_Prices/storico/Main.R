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
source("./ausiliaryFunctions/whatAreFactors.R")
source("./ausiliaryFunctions/scaleNumericalData.R")
source("./ausiliaryFunctions/percentageNumCol.R")
source("./ausiliaryFunctions/solveSkewness.R")
source("./ausiliaryFunctions/removeNumericalCorrelation.R")

trainset<-read.csv2("./input/train.csv", sep =",")
testset<-read.csv2("./input/test.csv", sep =",")

trainset<-trainset %>% select(-Id)
SalePrice<-trainset$SalePrice

# trainset will not have the target variable!
trainset<-trainset %>% select(-SalePrice)

skimmed <- skim_to_wide(trainset)
skimmed[, c(1:4, 8:9, 13, 15:15)] # we see skewness in our numerical data

namesFactor<-whatAreFactors(trainset) # to mantain memory after one hot encounding what are the factory variables
percentageNumCol(trainset) # to see how if I will remove to many columns


##############################  ONE-HOT ENCODING ##################################

# One-Hot Encoding
# Creating dummy variables is converting a categorical variable to as many binary variables as here are categories.
dummies_modelAllColumns <- dummyVars( ~ ., data=trainset)
trainData_matAllColumns <- predict(dummies_modelAllColumns, newdata = trainset)
#dim(trainData_matAllColumns)

dummies_modelFullRank <- dummyVars( ~ ., data=trainset, fullRank = TRUE)
trainData_matFullRank <- predict(dummies_modelFullRank, newdata = trainset)
#dim(trainData_matFullRank)

##############################  FILL THE NA VALUES ##################################

method <- "julien"
trainsetRegAllColumns<-data.frame(trainData_matAllColumns)
trainsetRegFullRank<-data.frame(trainData_matFullRank)


# trainsetRegAllColumns<- dealingWithNA(trainsetRegAllColumns, "julien", ncol(trainset), namesFactor)
# saveRDS(trainsetRegAllColumns,"./dataSaved/trainsetRegAllColumns.RDS")
trainsetRegAllColumns<-readRDS("./dataSaved/trainsetRegAllColumns.RDS")

# trainsetRegFullRank<- dealingWithNA(trainsetRegFullRank, "julien", ncol(trainset), namesFactor)
# saveRDS(trainsetRegFullRank,"./dataSaved/trainsetRegFullRank.RDS")
trainsetRegFullRank<-readRDS("./dataSaved/trainsetRegAllColumns.RDS")

# at this point i've the columns with NA filled and the oldest one with near zero variance

##############################  SOLVING PROBLEM OF SKEWNESS ##################################

par(mfrow=c(1,1))
#hist(SalePrice)
responseSkew <- BoxCoxTrans(SalePrice)
SalePriceTrans <- predict(responseSkew, SalePrice)
#hist(SalePriceTrans)

trainsetRegFullRankSym<-solveSkewness(trainsetRegFullRank)
trainsetRegAllColumnsSym<-solveSkewness(trainsetRegAllColumns)


##############################  CORRELATIONS  ##############################  

# we remove predictors with too big correlations

trainsetRegFullRankSymNoCor<-removeNumericalCorrelation(trainsetRegFullRankSym)
trainsetRegAllColumnsSymNoCor<-removeNumericalCorrelation(trainsetRegAllColumnsSym)

##############################  SCALE NUMERICAL DATA ##############################  


trainsetRegFullRankSymNoCorScaled<-scaleNumericalData(trainsetRegFullRankSymNoCor)
trainsetRegAllColumnsSymNoCorScaled<-scaleNumericalData(trainsetRegAllColumnsSymNoCor)


##############################  DEFINING A MACHINE LEARNING MODEL  ##############################  

# we have three datasets as input:
# -trainsetRegAllColumns: with all columns after one hot encoding
# -trainsetRegFullRank: with full rank
# -trainsetRegFullRankSym: removed skewness with all columns
# -trainsetRegAllColumnsSym: removed skewness full rank
# -trainsetRegFullRankSymNoCorScaled:  removed skewness and correlation with all columns and scaled
# -trainsetRegAllColumnsSymNoCorScaled: removed skewness and correlation full rank and scaled

# we have two possible outputs:
# -SalePrice
# -SalePriceTrans

##############################  CREATING DATA BASE  ##############################  

trainingIndex<-createDataPartition(SalePrice, p = 0.8, list = FALSE)

SalePriceTransTrain<-SalePriceTrans[trainingIndex]

training1<-trainsetRegAllColumns[trainingIndex,]
training2<-trainsetRegFullRank[trainingIndex,]
training3<-trainsetRegFullRankSym[trainingIndex,]
training4<-trainsetRegAllColumnsSym[trainingIndex,]
training5<-trainsetRegFullRankSymNoCorScaled[trainingIndex,]
training6<-trainsetRegAllColumnsSymNoCorScaled[trainingIndex,]


dummies_model <- dummyVars(~ ., data=training1)
training1_mat <- predict(dummies_model, newdata = training1)
dummies_model <- dummyVars(~ ., data=training2)
training2_mat <- predict(dummies_model, newdata = training2)
dummies_model <- dummyVars(~ ., data=training3)
training3_mat <- predict(dummies_model, newdata = training3)
dummies_model <- dummyVars(~ ., data=training4)
training4_mat <- predict(dummies_model, newdata = training4)
dummies_model <- dummyVars(~ ., data=training5)
training5_mat <- predict(dummies_model, newdata = training5)
dummies_model <- dummyVars(~ ., data=training6)
training6_mat <- predict(dummies_model, newdata = training6)

# Define the training control
fitControl <- trainControl(
  method = 'cv',                   # k-fold cross validation
  number = 5                    # number of folds
) 

######## linear model -----

trainingTransNoCorScaled<-trainData_mat[trainingIndex,]
SalePriceTransTrain<-SalePriceTrans[trainingIndex]

dummies_model <- dummyVars(~ ., data=trainingTransNoCorScaled, fullRank = TRUE)
# Create the dummy variables using predict. The Y variable will not be present in trainData_mat.
# The Y variable (SalePrice) will not be present in trainData_mat.
trainData_mat <- predict(dummies_model, newdata = trainingTransNoCorScaled)

model <- train(trainData_mat, SalePriceTransTrain,
               method = "lm",
               trControl = fitControl
               )

######## random forest -----

model_rf6 <- train(training6_mat, SalePriceTransTrain,
                  method='rf',
                  tuneLength=5, 
                  trControl = fitControl)
saveRDS(model_rf6,"./dataSaved/randomForest/model_rf.RDS")
plot(model_rf)

########  SVM -----

trainingTransNoCorScaled<-trainsetTransNoCorScaled[trainingIndex,]
SalePriceTransTrain<-SalePriceTrans[trainingIndex]

nzv <- nearZeroVar(trainingTransNoCorScaled, saveMetrics = TRUE)
# this is the case in which we have very low variance and possibles NA values that we 
# can't treat with knn 
trainingTransNoCorScaledNovar<- trainingTransNoCorScaled[, !nzv[,"nzv"]]
dummies_model <- dummyVars(~ ., data=trainingTransNoCorScaledNovar)
trainData_mat <- predict(dummies_model, newdata = trainingTransNoCorScaledNovar)

svmRTuned <- train(trainData_mat, SalePriceTransTrain,
                   method = "svmRadial",
                   tuneLength = 14,
                   trControl = fitControl
                   )

plot(svmRTuned)

######## XGBOOST -----
set.seed(123)

dummies_model <- dummyVars(~ ., data=trainsetReg)
# Create the dummy variables using predict. The Y variable will not be present in trainData_mat.
# The Y variable (SalePrice) will not be present in trainData_mat.
trainData_mat <- predict(dummies_model, newdata = trainsetReg)

xgb.fit1 <- xgb.cv(
  data = trainData_mat,
  label = SalePrice,
  nrounds = 1000,
  nfold = 5,
  objective = "reg:linear",  # for regression models
  verbose = 0,               # silent,
  early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
)

ggplot(xgb.fit1$evaluation_log) +
  geom_line(aes(iter, train_rmse_mean), color = "red") +
  geom_line(aes(iter, test_rmse_mean), color = "blue")



# create hyperparameter grid
hyper_grid <- expand.grid(
  eta = c(.01, .05, .1, .3),
  max_depth = c(1, 3, 5, 7),
  min_child_weight = c(1, 3, 5, 7),
  subsample = c(.65, .8, 1), 
  colsample_bytree = c(.8, .9, 1),
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0                     # a place to dump results
)


# grid search 
for(i in 1:nrow(hyper_grid)) {
  
  # create parameter list
  params <- list(
    eta = hyper_grid$eta[i],
    max_depth = hyper_grid$max_depth[i],
    min_child_weight = hyper_grid$min_child_weight[i],
    subsample = hyper_grid$subsample[i],
    colsample_bytree = hyper_grid$colsample_bytree[i]
  )
  
  # reproducibility
  set.seed(123)
  
  # train model
  xgb.tune <- xgb.cv(
    params = params,
    data = trainData_mat,
    label = SalePrice,
    nrounds = 5000,
    nfold = 5,
    objective = "reg:linear",  # for regression models
    verbose = 0,               # silent,
    early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(xgb.tune$evaluation_log$test_rmse_mean)
  hyper_grid$min_RMSE[i] <- min(xgb.tune$evaluation_log$test_rmse_mean)
}
saveRDS(xgb.tune,"xgb.tune.RDS")
hyper_grid %>%
  dplyr::arrange(min_RMSE) %>%
  head(10)


# parameter list
params <- list(
  eta = 0.1,
  max_depth = 3,
  min_child_weight = 1,
  subsample = 1,
  colsample_bytree = 1
)
# train final model
xgb.fit.final <- xgboost(
  params = params,
  data = trainData_mat,
  label = SalePrice,
  #nfold = 5,
  nrounds = 1576,
  objective = "reg:linear",
  verbose = 0
)
saveRDS(xgb.fit.final,"xgb.fit.final.RDS")
# # plot error vs number trees
# ggplot(xgb.fit.final$evaluation_log) +
#   geom_line(aes(iter, train_rmse_mean), color = "red") +
#   geom_line(aes(iter, test_rmse_mean), color = "blue")

# create importance matrix
importance_matrix <- xgb.importance(model = xgb.fit.final)

# variable importance plot
xgb.plot.importance(importance_matrix, top_n = 10, measure = "Gain")


######## Neural Network ----

library(nnet)
# we train a NNet with only one hidden layer
trainingTransNoCorScaled<-trainsetTransNoCorScaled[trainingIndex,]
SalePriceTransTrain<-SalePriceTrans[trainingIndex]

nzv <- nearZeroVar(trainingTransNoCorScaled, saveMetrics = TRUE)
# this is the case in which we have very low variance and possibles NA values that we 
# can't treat with knn 
trainingTransNoCorScaledNovar<- trainingTransNoCorScaled[, !nzv[,"nzv"]]

nnetGrid<- expand.grid( .decay = c(0, 0.01, .1),
                        .size = c(1:10),
                        .bag = FALSE # use bagging or different initial weights? 
                        )
ctrl <- trainControl(
  method = 'cv',                   # k-fold cross validation
  number = 5                     # number of folds
) 
set.seed(100)

library(doParallel)
cl <- makePSOCKcluster(5)
registerDoParallel(cl)

nnetTune <- train(trainingTransNoCorScaledNovar, SalePriceTransTrain,
                  method = "avNNet", # we train different nnet with different initial weights
                  tuneGrid = nnetGrid,
                  trControl = ctrl,
                  linout = TRUE, # we want linear output
                  trace = FALSE, # reduce of amout of printed out
                  MaxNWts = 10*(ncol(trainingTransNoCorScaledNovar)+1)+10 +1, # number of parameters
                  maxit = 500 # iteration for backpropagation
                  )
stopCluster(cl)
saveRDS(nnetTune,"nnetTune.RDS")
plot(nnetTune)

