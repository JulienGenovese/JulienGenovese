########################### ANALYZING IRIS DATASET ########################### 

### this script is to apply all the theory we have seen in the previous lectures

library(datasets)
library(tidyverse)
library(caret)
library(magrittr)
library(tictoc)

rm(list = ls())
cat("\014")

data(iris)
summary(iris)
attach(iris)

plot(iris, col = Species)
dataset <- select(iris, Sepal.Length, Sepal.Width, Species)

ggplot(dataset, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) +
  geom_point()

########################### WORKING ON THE DIFFERENT SPLIT ########################### 

set.seed(1)

# using simple random sample 

trainingRows <- sample(1:length(Species), size = .8 * length(Species))
dataset <- dataset %>% mutate(selected = 0)
dataset$selected[trainingRows] <- 1
dataset$selected %<>% as.factor
plot1 <- ggplot(dataset, aes(x = Sepal.Length, y = Sepal.Width, color = selected)) +
  geom_point()
table(dataset %>% .$Species)
table(dataset %>% filter(selected == 1) %>% .$Species)

# using stratified random sample

trainingRows <- createDataPartition(Species, 
                                    p = .8,
                                    list = FALSE)
dataset <- dataset %>% mutate(selected = 0)
dataset$selected[trainingRows] <- 1
dataset$selected %<>% as.factor
plot2 <- ggplot(dataset, aes(x = Sepal.Length, y = Sepal.Width, color = selected)) +
  geom_point()
table(dataset %>% .$Species)
table(dataset %>% filter(selected == 1) %>% .$Species)

# compare the two plots

plot1
plot2 # more dispersed 

dataset %<>% select(-selected)

########################### COMPARING RESAMPLING ON ML MODELS ########################### 

############# SVM ############# 

set.seed(100)

# repeated 10-fold cv

tic()
svmFitRep10F <- train(Species ~., 
                data = dataset,
                method = "svmRadial",
                preProc = c("center", "scale"),
                metric = "Accuracy",
                tuneLength = 10,
                trControl = trainControl(method = "repeatedcv",
                                         repeats = 10,
                                         classProbs = TRUE)
                )
timesvmFitRep10F <- toc()

# 10-fold cv

tic()
svmFit10F <- train(Species ~., 
                data = dataset,
                method = "svmRadial",
                preProc = c("center", "scale"),
                metric = "Accuracy",
                tuneLength = 10,
                trControl = trainControl(method = "cv",
                                         number = 10,
                                         classProbs = TRUE)
)
timesvmFit10F <- toc()

# bootstrap with 50/100 resampling

tic()
svmboot100 <- train(Species ~., 
                   data = dataset,
                   method = "svmRadial",
                   preProc = c("center", "scale"),
                   metric = "Accuracy",
                   tuneLength = 10,
                   trControl = trainControl(method = "boot",
                                            number = 100
                                            )
                  )
timesvmboot100 <-toc()

tic()
svmboot50 <- train(Species ~., 
                    data = dataset,
                    method = "svmRadial",
                    preProc = c("center", "scale"),
                    metric = "Accuracy",
                    tuneLength = 10,
                    trControl = trainControl(method = "boot",
                                             number = 50
                    )
)
timesvmboot50 <- toc()

tic()

svmLOOCV <- train(Species ~., 
                 data = dataset,
                 method = "svmRadial",
                 preProc = c("center", "scale"),
                 metric = "Accuracy",
                 tuneLength = 10,
                 trControl = trainControl(method = "LOOCV")
                  )
timesvmLOOCV <-toc()

timesvmFitRep10F$toc -timesvmFitRep10F$tic
timesvmFit10F$toc - timesvmFit10F$tic
timesvmboot100$toc - timesvmboot100$tic
timesvmboot50$toc - timesvmboot50$tic
timesvmLOOCV$toc - timesvmLOOCV$tic

svmFitRep10F
svmFit10F
svmboot100
svmboot50
svmLOOCV

plot(svmFitRep10F) # -> 100 validation errors
plot(svmFit10F) # -> 10 validation errors
plot(svmboot100) 
plot(svmboot50)
plot(svmLOOCV)


########################### COMPARING DIFFERENT ML MODELS ########################### 

set.seed(102)

knnFitRep <- train(Species ~., 
               data = dataset,
               method = "knn",
               metric = "Accuracy",
               preProc = c("center", "scale"),
               tuneLength = 30,
               trControl = trainControl(method = "repeatedcv",
                                        repeats = 10,
                                        classProbs = TRUE)
                )
plot(knnFitRep)

knnFitboot <- train(Species ~., 
                data = dataset,
                method = "knn",
                metric = "Accuracy",
                preProc = c("center", "scale"),
                tuneLength = 30,
                trControl = trainControl(method = "boot",
                                         number = 100
                )
)
plot(knnFitboot)

knnFitcv <- train(Species ~., 
                    data = dataset,
                    method = "knn",
                    metric = "Accuracy",
                    preProc = c("center", "scale"),
                    tuneLength = 30,
                    trControl = trainControl(method = "cv",
                                             number = 10,
                                             classProbs = TRUE)
)

plot(knnFitcv)

multinomLogRegFit <- train(Species ~., 
                data = dataset,
                method = "multinom",
                metric = "Accuracy",
                preProc = c("center", "scale"),
                tuneLength = 30,
                trace = FALSE,
                trControl = trainControl(method = "repeatedcv",
                                         repeats = 10,
                                         classProbs = TRUE)
)

plot(multinomLogRegFit)

svmFitRep10F <- train(Species ~., 
                      data = dataset,
                      method = "svmRadial",
                      preProc = c("center", "scale"),
                      metric = "Accuracy",
                      tuneLength = 15,
                      trControl = trainControl(method = "repeatedcv",
                                               repeats = 10,
                                               classProbs = TRUE)
)

plot(svmFitRep10F)

# comparing models

resamp <- resamples(list(KNN = knnFitRep,
                        svm = svmFitRep10F,
                        multipleLog = multinomLogRegFit)
                   )
summary(resamp)
bwplot(resamp)

