# loading libraries
library(dplyr)
library(randomForest)
library(forecast)
library(ggplot2)
library(stringr)

###load the dataset  ----
cat("\014")
rm(list = ls())
trainSet<-read.csv2("./../input/train.csv",sep =",",stringsAsFactors = FALSE)
testSet<-read.csv2("./../input/test.csv",sep=",")
colnames(trainSet)

###### feature engineering ----



learn3<-trainSet %>% select(c("Survived","Embarked")) %>% group_by(Embarked)
summarise(learn3, sumSurvived = sum(Survived, na.rm = TRUE)/n())

# we see that the embarked port seeems not to be very important to survive. 
# it makes sense

learn4<-trainSet %>% select(c("Survived","Pclass")) %>% group_by(Pclass)
summarise(learn4, sumSurvived = sum(Survived, na.rm = TRUE)/n())
# on the contrary the Pclass seems to bee important at first sight
t1<-trainSet %>% mutate(isCabin = !(Cabin == "")) %>% group_by(isCabin)
summarise(t1, sumSurvived = sum(Survived, na.rm = TRUE)/n())
# for the moment also isCabin seems to be important


trainSet$Title<-unlist(lapply(strsplit(trainSet$Name,split=",") ,FUN=  function(name) {
   str_trim(strsplit(name[2],split="\\.")[[1]][1])
   
   }
   ))

# try to work only with a part of the dataset


X_train<-trainSet %>% mutate(isCabin = !(Cabin == "")) %>%
   select(-c("PassengerId","Name","Ticket","Cabin"))

# converting data type ----

X_train$Sex<- X_train$Sex %>% as.factor
X_train$Pclass<- X_train$Pclass %>% as.factor
X_train$Embarked<- X_train$Embarked %>% as.factor
X_train$Age<-X_train$Age %>% as.numeric
X_train$Survived <- X_train$Survived %>% as.factor
X_train$Fare<-X_train$Fare %>% as.numeric
X_train$Title<-X_train$Title %>% as.factor
###
summary(X_train)
colSums(is.na(X_train)) 
# percentage of na in Age and Embarked
trainGrouped <- X_train %>% group_by(Sex,Pclass)
summarise(trainGrouped, meanAge = mean(Age, na.rm = TRUE) , meadianAge = median(Age, na.rm = TRUE),numberOfNA_Age = sum(is.na(Age))/nrow(X_train),)

sum(is.na(X_train$Age))/sum(nrow(X_train))
sum(is.na(X_train$Embarked))/sum(nrow(X_train))
learning<-X_train[1:600,]
testing<-X_train[600:891,]
##### SEE DISTRIBUTION AGE

histAgeBeforeFill<-ggplot(X_train[!is.na(X_train$Age),], aes(x=Age)) + 
   geom_histogram(aes(y=..density..), colour="black", fill="white")+
   geom_density(alpha=.2, fill="#FF6666") 
histAgeBeforeFill

methodFillNA_Age<-"mean"
summaryBefFill<-summary(X_train$Age)
# 1) FIRST METHOD:  filling NA Values WITH MEAN----
if(methodFillNA_Age =="mean"){
   X_train$Age[is.na(X_train$Age)]<- mean(X_train$Age[!is.na(X_train$Age)]) 
   summaryMeanFill<-summary(X_train$Age)
   histAgeMeanFill<-ggplot(X_train, aes(x=Age)) + 
      geom_histogram(aes(y=..density..), colour="black", fill="white")+
      geom_density(alpha=.2, fill="#FF6666") 
   histAgeMeanFill
}

#### 2) SECOND METHOD OF FILLING NA -----

# filling NA Values ----
if(methodFillNA_Age=="meanGroup"){
   qplot(Pclass,Age, data=X_train, geom=c("boxplot"),
      fill=Pclass, main="Age by Category",
      xlab="", ylab="Age")


   qplot(Sex,Age, data=X_train, geom=c("boxplot"),
      fill=Pclass, main="Age by Category",
      xlab="", ylab="Age")
   qplot(Embarked,Age, data=X_train, geom=c("boxplot"),
      fill=Embarked, main="Age by Category",
      xlab="", ylab="Age")
 
   trainGrouped <- X_train %>% group_by(Sex,Pclass)
   summarise(trainGrouped, meanAge = mean(Age, na.rm = TRUE) , meadianAge = median(Age, na.rm = TRUE),numberOfNA = sum(is.na(Age)))

   meanGroup <-unlist(group_map(trainGrouped[!is.na(trainGrouped$Age),], ~median(.x$Age)))
   X_train$Age[is.na(X_train$Age) & X_train$Sex == "female" & X_train$Pclass=="1"]<- meanGroup[1]
   X_train$Age[is.na(X_train$Age) & X_train$Sex == "female" & X_train$Pclass=="2"]<- meanGroup[2]
   X_train$Age[is.na(X_train$Age) & X_train$Sex == "female" & X_train$Pclass=="3"]<- meanGroup[3]
   X_train$Age[is.na(X_train$Age) & X_train$Sex == "male" & X_train$Pclass=="1"]<- meanGroup[4]
   X_train$Age[is.na(X_train$Age) & X_train$Sex == "male" & X_train$Pclass=="2"]<- meanGroup[5]
   X_train$Age[is.na(X_train$Age) & X_train$Sex == "male" & X_train$Pclass=="3"]<- meanGroup[6]

   summaryMeanGroupFill<-summary(X_train$Age)
   sum(is.na(X_train$Age))/sum(nrow(X_train))
   sum(is.na(X_train$Embarked))/sum(nrow(X_train))

   histAgeMeanGroupFill<-ggplot(X_train, aes(x=Age)) + 
      geom_histogram(aes(y=..density..), colour="black", fill="white")+
      geom_density(alpha=.2, fill="#FF6666") 
   histAgeMeanGroupFill
}



### MODEL SELECTION----

X_train<-X_train %>% select(-c("Embarked","SibSp","Parch","isCabin"))
#X_train<-X_train %>% select(-c("isCabin"))

set.seed(2)

# k fold for estimating test error

n_folds <- 10 # k = 10
n_train<- nrow(X_train)
folds_i <- sample(rep(1:n_folds, length.out = n_train))
ntree<-seq(10,500,20)
fitted_models <- apply(t(ntree), 2, function(par) randomForest(Survived~. ,data = X_train, ntree = par))

# we compute the train error using the confusion matrix
train_error <- sapply(fitted_models, function(obj) (obj$confusion[2]+obj$confusion[3])/nrow(X_train))
# let's compute the validation error
cv_tmp <- matrix(NA, nrow = n_folds, ncol = length(ntree))
for (k in 1:n_folds) {
   # here we select the validation set
   test_i <- which(folds_i == k) 
   train_xy <- X_train[-test_i, ]
   test_xy <- X_train[test_i, ]
   fitted_models <- apply(t(ntree), 2, function(par) randomForest(Survived~. ,data = train_xy, ntree = par))
   x <- test_xy %>% select(-Survived)
   y <- (test_xy %>% select(Survived))[[1]] %>% as.factor
   pred <- mapply(function(obj) predict(obj, data.frame(x)), fitted_models)

   cv_tmp[k, ] <- sapply(as.list(data.frame(pred)), function(y_hat) length(which(y_hat != y))/length(y))

}
cv <- colMeans(cv_tmp)
accuracyMin<-1-min(cv)
print(accuracyMin)
print(fitted_models[[which.min(cv)]])
importance(fitted_models[[which.min(cv)]])
plot(ntree, train_error, type = "l", lwd = 2, col = gray(0.4), ylab = "Training error", 
     xlab = "Number of trees", main = paste0(n_folds,"-fold Cross-Validation"), ylim = c(0.1, 0.8))
lines(ntree, cv, lwd = 2, col = "steelblue2")
points(ntree, cv, col = "steelblue2", pch = 19)
legend(x = "topright", legend = c("Training error", "Cross-validation error"), 
       lty = c(1, 2, 1), lwd = rep(2, 3), col = c(gray(0.4), "darkred", "steelblue2"), 
       text.width = 0.2, cex = 0.85)


