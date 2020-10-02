##############################  Analyzing HOUSE PRICES DATA

##############################  LIBRARIES  AND IMPORT DATA ##################################

library(caret)
library(doMC)
library(tidyverse)
library(xgboost)
library(magrittr)


rm(list = ls())
cat("\014")

source("knnImputing.R")

trainset <- read.csv2("./input/train.csv",
                      sep = ",", stringsAsFactors = FALSE
                      )
testset <-  read.csv2("./input/test.csv",
                      sep = ",", stringsAsFactors = FALSE
                      )
trainX <- trainset %>% select(-c(SalePrice,Id))
trainY <- trainset %>% select(SalePrice)
testX <- testset %>% select(-Id)

##############################  PREPOCESSING THE DATA 

# trying to remove skewness from SalePrice

SalePriceTrans <- BoxCoxTrans(trainY$SalePrice)
hist(predict(SalePriceTrans, trainY$SalePrice), xlab = "SalePriceTransformed", main = "BoxCoxTransformation")
transTrainY <- predict(SalePriceTrans, trainY$SalePrice)

##############################  STUDY NA VALUES ##############################  

percNa <- colSums(is.na(trainX))/nrow(trainX)
percaNAoverTh <- percNa[percNa > .0]
NAvalues <- data.frame(name = names(percaNAoverTh), val = percaNAoverTh)
ggplot(data = NAvalues, aes(x=name, y = val )) +
  geom_bar(stat="identity") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

################
# we can see that for Alley, fence, MiscFeature, PoolQC, NA is a value, not a 
# missing value

unique(trainX$Fireplaces) # we can see that 0 is present so this NA is real

# lot frontage is also a NA
# we see from the description file that some columns have ficticious NA

dataset <- bind_rows(trainX, testX)

dataset$Alley[is.na(dataset$Alley)] <- "No alley access"
dataset$BsmtQual[is.na(dataset$BsmtQual)] <- "No basement"
dataset$BsmtCond[is.na(dataset$BsmtCond)] <- "No basement"
dataset$BsmtExposure[is.na(dataset$BsmtExposure)] <- "No basement"
dataset$BsmtFinType1[is.na(dataset$BsmtFinType1)] <- "No basement"
dataset$BsmtFinType2[is.na(dataset$BsmtFinType2)] <- "No basement"
dataset$FireplaceQu[is.na(dataset$FireplaceQu)] <- "No fireplace"
dataset$GarageType[is.na(dataset$GarageType)] <- "No garage"
dataset$GarageFinish[is.na(dataset$GarageFinish)] <- "No garage"
dataset$GarageQual[is.na(dataset$GarageQual)] <- "No garage"
dataset$GarageCond[is.na(dataset$GarageCond)] <- "No garage"
dataset$PoolQC[is.na(dataset$PoolQC)] <- "No pool"
dataset$Fence[is.na(dataset$Fence)] <- "No fence"
dataset$MiscFeature[is.na(dataset$MiscFeature)] <- "None"

percNa <- colSums(is.na(dataset))/nrow(dataset)
percaNAoverTh <- percNa[percNa > .0]
NAvalues <- data.frame(name = names(percaNAoverTh), val = percaNAoverTh)

# we plot again the distribution of NA 
# we can see now that we don't have a lot of NA in general
ggplot(data = NAvalues, aes(x=name, y = val )) +
  geom_bar(stat="identity") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

##############################   we transform in factors the categorichal columns ##############################  

dataset <- modify_if(dataset, is.character, as.factor)
# we can see that these other variables are numbers but are associated to a factor level
dataset$OverallQual %<>% as.factor() 
dataset$OverallCond %<>% as.factor() 
dataset$MSSubClass %<>% as.factor() 

###############################  NZV ##############################  

# we remove for the moment the columns with nzv because can cause problem in the knn 
nzr <- nearZeroVar(dataset, saveMetrics = TRUE)

datasetwithOutNzv <- dataset[,!nzr$nzv]

##############################  NA FILLING ##############################  

registerDoMC(cores = 5)
trainFilled <- fillingByKNN(datasetwithOutNzv)

saveRDS(trainFilled, "trainFilled.RDS")
# we want to analyze the differences in the distribution between the filled dataset
# and the previous one.

# LotFrontage  

dat <- data.frame(cond = factor(rep(c("NA","NoNA"), each= nrow(dataset))), 
                  LotFrontage = c(dataset$LotFrontage,trainFilled$LotFrontage))
cdat <- plyr::ddply(dat  %>% filter(!is.na(LotFrontage)) , "cond", summarise, LotFrontage.mean = mean(LotFrontage))

dat %>% 
  filter(!is.na(LotFrontage)) %>% 
  ggplot(aes(x = LotFrontage, colour = cond)) + geom_density() + 
  geom_vline(data = cdat, aes(xintercept = LotFrontage.mean,  colour=cond),
           linetype="dashed", size=1)

# GarageYrBlt 

dat <- data.frame(cond = factor(rep(c("NA","NoNA"), each = nrow(dataset))), 
                  GarageYrBlt = c(dataset$GarageYrBlt,trainFilled$GarageYrBlt))
cdat <- plyr::ddply(dat  %>% filter(!is.na(GarageYrBlt)) , "cond", summarise, GarageYrBlt.mean = mean(GarageYrBlt))

dat %>% 
  filter(!is.na(GarageYrBlt)) %>% 
  ggplot(aes(x = GarageYrBlt, colour = cond)) + geom_density() + 
  geom_vline(data = cdat, aes(xintercept = GarageYrBlt.mean,  colour=cond),
             linetype = "dashed", size=1)
# we can see that the columns with the majority of NA have the same distribution of the
# data after the filling 

## Now We have to come back to the variables with also the Nzv values

trainFilledAllcol <- trainFilled %>% bind_cols(dataset[,nzr$nzv] )
toRemove <- colnames(trainFilledAllcol)[colSums(is.na(trainFilledAllcol)) > 0 ]
trainFilledAllcol <- trainFilledAllcol %>% select(-toRemove)   

##################### MODELING WITH A MACHINE LEARNING MODEL ##################### 

trainFilledAllcol <- readRDS("trainFilledAllcol.RDS")

learningSetX <- trainFilledAllcol[1:nrow(trainX), ]
learningSetY <- transTrainY
trainingRows <- createDataPartition(learningSetY , p = .8, list = FALSE)

dfTrainX <- learningSetX[trainingRows,]
dfTrainY <- learningSetY[trainingRows]
dfTestX <- learningSetX[-trainingRows,]
dfTestY <- learningSetY[-trainingRows]

testingSet <- trainFilledAllcol[(nrow(trainX) + 1):nrow(trainFilledAllcol), ]

ctrl <- trainControl(method = "repeatedcv", 
                     number = 10,
                     repeats = 3
                     )
# LINEAR MODELS 

dummies_modelLINEAR <- dummyVars( ~ ., data = dfTrainX  , fullRank = TRUE)
data_matFullRankLinear <- predict(dummies_modelLINEAR , newdata = dfTrainX )
nzrLinear <- nearZeroVar(data_matFullRankLinear, saveMetrics = TRUE)
data_matFullRankLinear <- data_matFullRankLinear[,!nzrLinear$nzv]

data_matLinearTest <- predict(dummies_modelLINEAR , newdata = dfTestX )
data_matLinearTest <- data_matLinearTest[,!nzrLinear$nzv]


# CLASSICAL LINEAR REGRESSION

set.seed(101)
registerDoMC(cores = 5)
lmFit <- train(y = dfTrainY,
               x = data_matFullRankLinear,
               method = "lm",
               trControl = ctrl)

# PLS 

set.seed(101)
registerDoMC(cores = 5)
plsTune <- train(y = dfTrainY,
                 x = data_matFullRankLinear,
                 method = "pls",
                 tuneLength = 100,
                 trControl = ctrl,
                 preProc = c("center", "scale")
)
plot(plsTune)

# principal component regression 

set.seed(101)
registerDoMC(cores = 5)
lmFitPCA <- train(
  y = dfTrainY,
  x = data_matFullRankLinear,
  method = "pcr",
  preProcess = c("center", "scale"),
  trControl = ctrl,
  tuneLength = 100
)
plot(lmFitPCA)

# ELASTIC NET

enetGrid <- expand.grid(.lambda = c(0, 0.01, .1),
                        .fraction = seq(.05, 1, length = 20)
)

set.seed(101)
registerDoMC(cores = 5)
enetTune <- train(y = dfTrainY,
                  x = data_matFullRankLinear,
                  method = "enet",
                  tuneGrid = enetGrid,
                  trControl = ctrl,
                  preProc = c("center", "scale")
)
plot(enetTune)

# Random forest

set.seed(101)
registerDoMC(cores = 5)
rf <- train(y = dfTrainY,
            x = dfTrainX,
            method = 'rf',
            tuneLength = 10, 
            trControl = ctrl
            )
plot(rf)
plot(varImp(rf))

########  XGBOOST -----

dummies_modelXGBOOST <- dummyVars( ~ ., data = dfTrainX  , fullRank = TRUE)
data_matFullRankXGBOOST <- predict(dummies_modelXGBOOST , newdata = dfTrainX )

# different part of tuningof xgboost : number of trees
fitControl <- trainControl(
  method = 'cv',                   # k-fold cross validation
  number = 5,                   # number of folds
  allowParallel = TRUE
) 
hyper_grid <- expand.grid(
  nrounds =  seq(from = 200, to = 1000, by = 50),
  eta = .1,
  gamma = 0,
  max_depth = 5,
  min_child_weight = 1,
  subsample = .8, 
  colsample_bytree = 0.8
)

registerDoMC(cores = 5)
xgb.fit1 <- train(y = dfTrainY,
                  x = data_matFullRankXGBOOST,
                  method = "xgbTree",
                  trControl = fitControl,
                  tuneGrid = hyper_grid,
                  verbose = TRUE,
                  nthread = 4
)
plot(xgb.fit1)
saveRDS(xgb.fit1,"./models/xgb.fit1.RDS")
readRDS("./models/xgb.fit1.RDS") -> xgb.fit1
hyper_grid <- expand.grid(
  nrounds =  xgb.fit1$bestTune$nrounds,
  eta = .1,
  gamma = 0,
  max_depth = c(2,3,10),
  min_child_weight = c(1,2,6),
  subsample = .8, 
  colsample_bytree = 0.8
)

registerDoMC(cores = 5)
xgb.fit2 <- train(y = dfTrainY,
                  x = data_matFullRankXGBOOST,
                  method = "xgbTree",
                  trControl= fitControl,
                  tuneGrid = hyper_grid,
                  verbose = TRUE,
                  nthread = 4
)
plot(xgb.fit2)
saveRDS(xgb.fit2,"./models/xgb.fit2.RDS")
readRDS("./models/xgb.fit2.RDS") -> xgb.fit2
hyper_grid <- expand.grid(
  nrounds =  xgb.fit1$bestTune$nrounds,
  eta = .1,
  gamma = seq(1/10,1,1/10),
  max_depth = xgb.fit2$bestTune$max_depth,
  min_child_weight = xgb.fit2$bestTune$min_child_weight,
  subsample = .8, 
  colsample_bytree = 0.8
)

xgb.fit3 <- train(y = dfTrainY,
                  x = data_matFullRankXGBOOST,
                  method = "xgbTree",
                  trControl= fitControl,
                  tuneGrid = hyper_grid,
                  verbose = TRUE,
                  nthread = 4
)
plot(xgb.fit3)
saveRDS(xgb.fit3,"./models/xgb.fit3.RDS")
readRDS("./models/xgb.fit3.RDS") -> xgb.fit3

hyper_grid <- expand.grid(
  nrounds =  xgb.fit1$bestTune$nrounds,
  eta = .1,
  gamma = xgb.fit3$bestTune$gamma,
  max_depth = xgb.fit2$bestTune$max_depth,
  min_child_weight = xgb.fit2$bestTune$min_child_weight,
  subsample = seq(6/10,1., 1/10), 
  colsample_bytree = seq(6/10, 1., 1/10)
)
#registerDoMC(cores = 5)
xgb.fit4 <- train(y = dfTrainY,
                  x = data_matFullRankXGBOOST,
                  method = "xgbTree",
                  trControl= fitControl,
                  tuneGrid = hyper_grid,
                  verbose = TRUE
)
plot(xgb.fit4)
saveRDS(xgb.fit4,"./models/xgb.fit4.RDS")

hyper_grid <- expand.grid(
  nrounds =  5000,
  eta = 0.01,
  gamma = xgb.fit3$bestTune$gamma,
  max_depth = xgb.fit2$bestTune$max_depth,
  min_child_weight = xgb.fit2$bestTune$min_child_weight,
  subsample = xgb.fit4$bestTune$subsample, 
  colsample_bytree = xgb.fit4$bestTune$colsample_bytree
)
registerDoMC(cores = 5)
xgb.fit5 <- train(y = dfTrainY,
                  x = data_matFullRankXGBOOST,
                  method = "xgbTree",
                  trControl= fitControl,
                  tuneGrid = hyper_grid,
                  verbose = TRUE,
                  nthread = 4
)
plot(xgb.fit5)
saveRDS(xgb.fit5,"./models/xgb.fit5.RDS")


## my final model 

hyper_grid <- expand.grid(
  nrounds =  5000,
  eta = 0.01,
  gamma = 0.1,
  max_depth = 3,
  min_child_weight = 1,
  subsample = 0.6, 
  colsample_bytree = 1
)

set.seed(101)
registerDoMC(cores = 5)
xgb.fitFinal <- train(
                  y = dfTrainY,
                  x = data_matFullRankXGBOOST, 
                  method = "xgbTree",
                  trControl = ctrl,
                  tuneGrid = hyper_grid,
                  verbose = TRUE
)
saveRDS(xgb.fitFinal,"./models/xgb.fitFinal.RDS")


##################### SELECTING THE MODEL ##################### 

lmFit <- readRDS("./models/lmFit.RDS")
lmFitPCA <- readRDS("./models/lmFitPCA.RDS")
plsTune <- readRDS("./models/plsTune.RDS")
enetTune <- readRDS("./models/enetTune.RDS")
rf <- readRDS("./models/rf.RDS")
xgb.fitFinal <- readRDS("./models/xgb.fitFinal.RDS")

resamp <- resamples(list(linear = lmFit, 
                         linearPCA = lmFitPCA,
                         linearePLS = plsTune,
                         elasticNet = enetTune,
                         randomForest = rf,
                         xgboost = xgb.fitFinal
                         )
                    )
bwplot(resamp)


##################### ANALYSIS OF THE MODEL ##################### 

matdfTestX <- predict(dummies_modelXGBOOST , newdata = dfTestX)
  
dfPredictedTestY <- predict(xgb.fitFinal, matdfTestX)
axisRange <- extendrange(c(exp(dfTestY), exp(dfPredictedTestY)))
plot(exp(dfTestY), exp(dfPredictedTestY),  
     ylim = axisRange,
     xlim = axisRange,
     xlab = "Observed",
     ylab = "Predicted")
abline(0, 1, col = "red", lty = 2)

##################### TRAINING OF THE FINAL MODEL ##################### 

matTrainX <- predict(dummies_modelXGBOOST , newdata = learningSetX)

hyper_grid <- expand.grid(
  nrounds =  5000,
  eta = 0.01,
  gamma = 0.1,
  max_depth = 3,
  min_child_weight = 1,
  subsample = 0.6, 
  colsample_bytree = 1
)

registerDoMC(cores = 5)
xgb.fitForPrediction <- train(
  y = learningSetY,
  x = matTrainX, 
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = hyper_grid,
  verbose = TRUE
)
saveRDS(xgb.fitForPrediction,"./models/xgb.fitForPrediction.RDS")


matdfSubX <- predict(dummies_modelXGBOOST , newdata = testingSet)
dfPredictedSubY <- predict(xgb.fitForPrediction, matdfSubX)

submission <- tibble(Id = testset$Id, SalePrice =  exp(dfPredictedSubY))
saveRDS(submission, "submission.RDS")
