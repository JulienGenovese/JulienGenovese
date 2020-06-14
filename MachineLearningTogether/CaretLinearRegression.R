# linear regression model
library(caret)# Simple linear regression model (lm means linear model)

data(mtcars)    # Load the dataset
head(mtcars)
# regression over a variable
model <- train(mpg ~ wt,
               data = mtcars,
               method = "lm") 

# Multiple linear regression model
model <- train(mpg ~ .,
               data = mtcars,
               method = "lm")
summary(model)
# Ridge regression model
model <- train(mpg ~ .,
               data = mtcars,
               method = "ridge") # Try using "lasso"
summary(model)

##########################  CROSS VALIDATION  #########################

## 10-fold CV# possible values: boot", "boot632", "cv", "repeatedcv", "LOOCV", "LGOCV"
fitControl <- trainControl(method = "repeatedcv",   
                           number = 10,     # number of folds
                           repeats = 10)    # repeated ten times

model.cv <- train(mpg ~ .,
                  data = mtcars,
                  method = "lasso",  # now we're using the lasso method
                  trControl = fitControl)  

model.cv   
# we add some preprocessing. We scale the data for the ridge regression 
model.cv <- train(mpg ~ .,
                  data = mtcars,
                  method = "lasso",
                  trControl = fitControl,
                  preProcess = c('scale', 'center')) # default: no pre-processing

model.cv

### we want to do some parameter tuning 

# Here I generate a dataframe with a column named lambda with 100 values that goes from 10^10 to 10^-2
lambdaGrid <- expand.grid(lambda = 10^seq(10, -2, length=100))

model.cv <- train(mpg ~ .,
                  data = mtcars,
                  method = "ridge",
                  trControl = fitControl,
                  preProcess = c('scale', 'center'),
                  tuneGrid = lambdaGrid,   # Test all the lambda values in the lambdaGrid dataframe
                  na.action = na.omit)   # Ignore NA values

model.cv 
ggplot(varImp(model.cv))
predictions <- predict(model.cv, mtcars)
predictions