################### ex. 6.1 pg. 137 ################### 

library(caret)
library(doMC)

rm(list = ls())
cat("\014")

## a) load the dataset

data(tecator)

?tecator

## b) PCA APPLICATION 

pcaObject <- prcomp(absorp, 
                    center = TRUE,
                    scale. = TRUE
                    )
percentVariance <- pcaObject$sd^2/sum(pcaObject$sd^2)*100
plot(percentVariance)
percentVariance[1:5] # i think that 1 is enough but it's only a linear method

varexp <- cumsum(percentVariance)
which(varexp >= 99.999999)
plot(varexp)

### 3) select from different models 

data <- data.frame(endpoints[,2], absorp)
names(data)[1] <- "fat"

hist(data$fat)

colSums(is.na(data))
# transBox <- BoxCoxTrans(data$fat)
# trans <- predict(trans, data$fat)
# hist(trans)

trainingRows <- createDataPartition(data$fat, p = .8, list = FALSE)
trainingData <- data[trainingRows, ]
testData <- data[-trainingRows,]
registerDoMC(cores = 5)

ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

set.seed(101)
lmFit <- train(fat ~ .,
               data = trainingData,
               method = "lm",
               trControl = ctrl)

set.seed(101)
lmFitPCA <- train(
  fat ~ .,
  data = trainingData,
  method = "pcr",
  preProcess = c("center", "scale"),
  trControl = ctrl,
  tuneLength = 100
)

set.seed(101)
rlmPCA <- train(fat ~ .,
                data = trainingData,
                method = "rlm",
                preProcess = "pca",
                trControl = ctrl
) 
set.seed(101)
plsTune <- train(fat ~ .,
                 data = trainingData,
                 method = "pls",
                 tuneLength = 200,
                 trControl = ctrl,
                 preProc = c("center", "scale")
)
varImp(plsTune)
enetGrid <- expand.grid(.lambda = c(0, 0.01, .1),
                        .fraction = seq(.05, 1, length = 20)
)
set.seed(101)
enetTune <- train(fat ~ .,
                  data = trainingData,
                  method = "enet",
                  tuneGrid = enetGrid,
                  trControl = ctrl,
                  preProc = c("center", "scale")
)

nnetGrid <- expand.grid(.decay = c(0., 0.01, .1), # regularization parameter
                        .size = c(1:10), # size for gradient descent
                        .bag = FALSE) # use bagging for resampling
set.seed(101)
nnetTune <- train(
  fat ~ .,
  data = trainingData,
  method = "avNNet",
  tuneGrid = nnetGrid,
  trControl = ctrl,
  preProc = c("center", "scale"), # center and scale 
  linout = TRUE, # we want a regression model
  trace = FALSE, # less output printed
  MaxNWts = 3 * (ncol(trainingData) + 1) + 3 + 1, # 3 hidden layers
  maxit = 500
)
plot(nnetTune)


resamp <- resamples(list(linear = lmFit, 
                         linearPCA = lmFitPCA,
                         linearePLS = plsTune,
                         linearRobust = rlmPCA,
                         elasticNet = enetTune))
                         #neuralNet = nnetTune))
bwplot(resamp)

resamp <- resamples(list(
                         linearePLS = plsTune,
                         elasticNet = enetTune,
                         neuralNet = nnetTune))
bwplot(resamp)

# I would choose the PLS because it's the easier model and is very good for 
# correlated models

################### ex. 6.2 pg. 138 ################### 

.rs.restartR()

library(caret)
library(doMC)

rm(list = ls())
cat("\014")

## a) download data

library(AppliedPredictiveModeling)
data(permeability)
dim(fingerprints)

## b) filtering data

nzr <- nearZeroVar(fingerprints)

fingerprintsNoVar <- fingerprints[,-nzr]
dim(fingerprintsNoVar)

## c) pls prediction

data <- data.frame(permeability,fingerprintsNoVar)
colnames(data)[1] <- "permeability"
transBox <- BoxCoxTrans(data$permeability)
trans <- predict(transBox, data$permeability)
hist(trans)
data$permeability <- trans

trainingRows <- createDataPartition(permeability, p = .8, list = FALSE)
trainingData <- data[trainingRows, ]
testData <- data[-trainingRows,]
registerDoMC(cores = 5)
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

set.seed(101)
plsTune <- train(permeability~ .,
                 data = trainingData,
                 method = "pls",
                 tuneLength = 50,
                 trControl = ctrl,
                 preProc = c("center", "scale")
)
plot(plsTune)

## d) pls prediction

prediction <- predict(plsTune, testData[,-1])
lmPls <- data.frame(obs = testData$permeability, pred = prediction)
defaultSummary(lmPls)

axisRange <- extendrange(c(lmPls$obs,lmPls$pred))

plot(lmPls$obs, lmPls$pred,  
     ylim = axisRange,
     xlim = axisRange)
abline(0, 1, col = "darkgrey", lty = 2)

## e) other models

set.seed(101)
lmFitPCA <- train(
  permeability~ .,
  data = trainingData,
  method = "pcr",
  preProcess = c("center", "scale"),
  trControl = ctrl,
  tuneLength = 100
)

set.seed(101)
rlmPCA <- train(permeability~ .,
                data = trainingData,
                method = "rlm",
                preProcess = "pca",
                trControl = ctrl
) 

enetGrid <- expand.grid(.lambda = c(0, 0.01, .1),
                        .fraction = seq(.05, 1, length = 20)
)
set.seed(101)
enetTune <- train(permeability~ .,
                  data = trainingData,
                  method = "enet",
                  tuneGrid = enetGrid,
                  trControl = ctrl,
                  preProc = c("center", "scale")
)
set.seed(101)
nnetGrid <- expand.grid(.decay = c(0., 0.01, .1), # regularization parameter
                        .size = c(1:10), # size for gradient descent
                        .bag = FALSE) # use bagging for resampling
nnetTune <- train(
  permeability~ .,
  data = trainingData,
  method = "avNNet",
  tuneGrid = nnetGrid,
  trControl = ctrl,
  preProc = c("center", "scale"), # center and scale 
  linout = TRUE, # we want a regression model
  trace = FALSE, # less output printed
  MaxNWts = 8 * (ncol(trainingData) + 1) + 8 + 1, # 3 hidden layers
  maxit = 500
)
resamp <- resamples(list(linearPCA = lmFitPCA,
                         linearePLS = plsTune,
                         linearRobust = rlmPCA,
                         elasticNet = enetTune))
bwplot(resamp)

################### ex. 6.3 pg. 139 ################### 

## a) load the data 

library(AppliedPredictiveModeling)
library(tidyverse)
library(caret)
library(doMC)

data(ChemicalManufacturingProcess)

## b) preprocess the data

predictors <- ChemicalManufacturingProcess %>% select(-Yield) 
yield <- ChemicalManufacturingProcess %>% select(Yield) %>% .$Yield

hist(yield)
# it's symmetric; we don't need Box-Box on this variable

colSums(is.na(predictors))/nrow(predictors)
# we need to inpute some NA values

set.seed(510)
trainingRows <- createDataPartition(yield,
                                    p = 0.7,
                                    list = FALSE)
trainPredictors <- predictors[trainingRows,]
trainYield <- yield[trainingRows]
testPredictors <- predictors[-trainingRows,]
testYield <- yield[-trainingRows]

#Pre-process trainPredictors and apply to trainPredictors and testPredictors
pp <- preProcess(trainPredictors, method = c("center","scale", "knnImpute"))
ppTrainPredictors <- predict(pp, trainPredictors)
ppTestPredictors <- predict(pp, testPredictors)

# let's see some correlation

library(corrplot)
correlations <- cor(ppTrainPredictors)

# c) train a model 

set.seed(101)
data <- data.frame(yield = trainYield, ppTrainPredictors)

registerDoMC(cores = 5)

ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

plsTune <- train(yield~ .,
                 data = data,
                 method = "pls",
                 tuneLength = 50,
                 trControl = ctrl,
                 preProc = c("center", "scale")
)
plot(plsTune)

# d) prediction on the test set 

prediction <- predict(plsTune, ppTestPredictors)

lmPls <- data.frame(obs = testYield, pred = prediction)
defaultSummary(lmPls)

# e) variable importance 

plot(varImp(plsTune))

# f) Explore the relationships 

# let's see the first three predictors 

ggplot(predictors,aes(y=yield,x=ManufacturingProcess32))+geom_point()+geom_smooth(method="lm")
ggplot(predictors,aes(y=yield,x=ManufacturingProcess09))+geom_point()+geom_smooth(method="lm")
ggplot(predictors,aes(y=yield,x=ManufacturingProcess13))+geom_point()+geom_smooth(method="lm")
