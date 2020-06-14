# Chapter 4 Lab: Logistic Regression
# The Stock Market Data

# we want to use logistic regression to predict if the market will go up or down
library(ISLR)
?Smarket

c<-Smarket
names(Smarket)
dim(Smarket)
summary(Smarket)
pairs(Smarket, col = Smarket$Direction)
cor(Smarket[,-9]) # very low correlation between variables. In finance it's quite normal
attach(Smarket)
plot(Volume)

# Logistic Regression

glm.fits=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,data=Smarket,family=binomial)
summary(glm.fits) # i want to see the informations associated to the model.
# very big p-values. They seems not very significant but it's quite normal in finance
coef(glm.fits)
summary(glm.fits)$coef
summary(glm.fits)$coef[,4]
glm.probs=predict(glm.fits,type="response") # we have the fitted probabilities
glm.probs[1:10]
glm.pred=rep("Down",1250)
glm.pred[glm.probs>.5]="Up"
table(glm.pred,Direction)
(507+145)/1250 # accuracy
mean(glm.pred==Direction) # same method for accuracy. In my opionion is like to flip a coin

# split in training data and validation data
train=(Year<2005)
Smarket.2005=Smarket[!train,]
dim(Smarket.2005)
Direction.2005=Direction[!train]
glm.fits=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,data=Smarket,family=binomial,subset=train)
glm.probs=predict(glm.fits,Smarket.2005,type="response")
glm.pred=rep("Down",252)
glm.pred[glm.probs>.5]="Up"
table(glm.pred,Direction.2005)
mean(glm.pred==Direction.2005)
mean(glm.pred!=Direction.2005)
glm.fits=glm(Direction~Lag1+Lag2,data=Smarket,family=binomial,subset=train)
glm.probs=predict(glm.fits,Smarket.2005,type="response") # we are predicting the market in 2005
glm.pred=rep("Down",252)
glm.pred[glm.probs>.5]="Up"
table(glm.pred,Direction.2005)
mean(glm.pred==Direction.2005) # our accuracy is a little bit better

predict(glm.fits,newdata=data.frame(Lag1=c(1.2,1.5),Lag2=c(1.1,-0.8)),type="response")




