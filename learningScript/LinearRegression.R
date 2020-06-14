# Chapter 3 Lab: Linear Regression

library(MASS) # some datasets we want to access
library(ISLR) # some other datasets

# Simple Linear Regression

fix(Boston)
names(Boston)
plot(medv~lstat,data=Boston)
lm.fit=lm(medv~lstat,data=Boston)
?Boston # to understand the dataset
attach(Boston)
fit1=lm(medv~lstat)
fit1
summary(fit1)
names(fit1)
coef(fit1)

plot(lstat,medv)
abline(fit1,lwd=3,col="red")

confint(fit1)
predict(fit1,data.frame(lstat=(c(5,10,15))), interval="confidence")

# Multiple Linear Regression

fit2=lm(medv~lstat+age,data=Boston)
summary(fit2)
fit3=lm(medv~.,data=Boston)
summary(fit3)
par(mfrow=c(2,2))
plot(fit3)

# Interaction Terms

fit5 = lm(medv~lstat*age,data=Boston)
summary(fit5)
# Non-linear Transformations of the Predictors
fit6=lm(medv~lstat+I(lstat^2))
summary(fit6)
par(mfrow =c(1,1))
plot(medv~lstat)
points(lstat,fitted(fit6), col= "red", pch= 20)
fit7=lm(medv~poly(lstat,4))
points(lstat,fitted(fit7), col= "blue", pch= 20) #it's seems that it's overfitting
summary(fit7)

# Qualitative Predictors

fix(Carseats)
names(Carseats)
summary(Carseats) # for numerical values you obtain some measures. 
# for chategorical ones you have a table summarizing the chategories
lm.fit=lm(Sales~.+Income:Advertising+Price:Age,data=Carseats)
summary(lm.fit)
attach(Carseats)
contrasts(ShelveLoc) # to see how R see the chategory

