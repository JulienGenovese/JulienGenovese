library(ISLR)
library(DAAG)

dfClean<-Hitters[!is.na(Hitters$Salary),]

# Multiple Linear Regression Example
fit <- lm(Salary ~ ., data=dfClean)
summary(fit) # show results

# Other useful functions
coefficients(fit) # model coefficients
confint(fit, level=0.95) # CIs for model parameters
fitted(fit) # predicted values
residuals(fit) # residuals
anova(fit) # anova table
vcov(fit) # covariance matrix for model parameters
influence(fit) # regression diagnostics 
layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page 
plot(fit)

# cross validation 

library(tidyverse)
library(caret)

# Load the data
data("swiss")
# Inspect the data
sample_n(swiss, 3)
# Split the data into training and test set
set.seed(123)
training.samples <- swiss$Fertility %>%
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- swiss[training.samples, ]
test.data <- swiss[-training.samples, ]
# Build the model
model <- lm(Fertility ~., data = train.data)
# Make predictions and compute the R2, RMSE and MAE
predictions <- model %>% predict(test.data)
data.frame( R2 = R2(predictions, test.data$Fertility),
            RMSE = RMSE(predictions, test.data$Fertility),
            MAE = MAE(predictions, test.data$Fertility))
# Define training control
train.control <- trainControl(method = "LOOCV")
# Train the model
model <- train(Fertility ~., data = swiss, method = "lm",
               trControl = train.control)
# Summarize the results
print(model)

# Define training control
set.seed(123) 
train.control <- trainControl(method = "cv", number = 10)
# Train the model
model <- train(Fertility ~., data = swiss, method = "lm",
               trControl = train.control)
# Summarize the results
print(model)

# Define training control
set.seed(123)
train.control <- trainControl(method = "repeatedcv", 
                              number = 10, repeats = 3)
# Train the model
model <- train(Fertility ~., data = swiss, method = "lm",
               trControl = train.control)
summary(model)
# Summarize the results
print(model)
predict(model$finalModel, train.data, interval = "confidence")
