# Analyzing Titanic data


###################################################################     Globals&CONSTANTS      ###############################################################
#INPUT PATH
input_path <- "C:/Users/Marco/Desktop/MLMilan/Titanic/data/"
#OUTPUT PATH
output_path <- "C:/Users/Marco/Desktop/MLMilan/Titanic/output/"
# Final file of predictions
output_name <- "Predictions"


###################################################################     LOAD FUNCTION      ###############################################################

# FUNCTION THAT CHECKS ALL THE LIBRARIES IF NOT INSTALLED WILL INSTALL ALL AND THEN WILL LOAD ALL LIBRARIES
pip <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}

###################################################################     LOAD LIBRARIES      ###############################################################

classical_packages = c("stringr","dplyr") # c("Matrix","lubridate","RMySQL","data.table","geosphere","lsa","tm")
pip(classical_packages)

model_packages = c("xgboost","randomForest","MLmetrics","caret","pROC")
pip(model_packages)


#################################################################      DEFINING FUNCTIONS     ##############################################################



#function that found which column as some 
# allmisscols <- sapply(data, function(x) all(is.na(x) | x == '' ))


# Return mode of a vector
statmod <- function(TX){
  TX <- table(as.vector(TX))
  if(length(TX)==1){
    return(names(TX)[1])
  } else{
    return(paste(names(TX)[TX == max(TX)], collapse="-"))
  }
}
# how to use:  statmod(X_train$Embarked)  or for each row: data$columnNew <- lapply(data$column, statmod)


###################################################################       LOADING DATA        ##############################################################


# sample data # library(bit64) #--> needed to avoid big integers
titanic_train  <- read.csv(file.path(input_path, "train.csv"), stringsAsFactors=FALSE)
titanic_test   <- read.csv(file.path(input_path, "test.csv"), stringsAsFactors=FALSE)


###################################################################     DATA EXPLORATION      ###############################################################

head(titanic_train)
summary(titanic_train)
summary(titanic_test)
str(titanic_train)

table(titanic_train$Survived) #balanced class problem
#see if there are differences in age
hist(titanic_train$Age)
hist(titanic_test$Age)
cor(titanic_train$Age[which(!is.na(titanic_train$Age))],titanic_train$Survived[which(!is.na(titanic_train$Age))]) #the higher the age, the less probability of surviving

#see if there are differences in Sex
table(titanic_train$Sex)
table(titanic_test$Sex)
table(titanic_train$Sex,titanic_train$Survived) #most survived people are females - significantly!

#see if there are differences in SibSp --> Number of Siblings/Spouses Aboard
table(titanic_train$SibSp)
table(titanic_test$SibSp)
cor(titanic_train$SibSp,titanic_train$Survived) #the higher the class, the less probability of surviving

#see if there are differences in Parch --> Number of Parents/Children Aboard
table(titanic_train$Parch)
table(titanic_test$Parch) #there are two strange 9
cor(titanic_train$Parch,titanic_train$Survived) #the higher the Parch, the higher probability of surviving

#see if there are differences in Fare
hist(log(titanic_train$Fare))
hist(log(titanic_test$Fare))
cor(titanic_train$Fare,titanic_train$Survived) #the higher the Fare, the higher probability of surviving
cor(log(titanic_train$Fare+0.001),titanic_train$Survived) #the higher the Fare, the higher probability of surviving

#see if there are differences in Cabin
#table(titanic_train$Cabin)
#table(titanic_test$Cabin)

#see if there are differences in Cabin
table(titanic_train$Embarked)
table(titanic_test$Embarked)
table(titanic_train$Embarked, titanic_train$Survived) # Class C has relative more probability of surviving



###################################################################     FEATURE ENGINEERING      ###############################################################
# it should be the same for train and test

# Creating a column named Name_Title --> it uses stringr
titanic_train['Name_Title'] = str_split_fixed(str_split_fixed(titanic_train$Name,", ",2)[,2]," ",2)[,1]
titanic_test['Name_Title'] = str_split_fixed(str_split_fixed(titanic_test$Name,", ",2)[,2]," ",2)[,1]
#grepl comes pretty useful here
titanic_train$Name_Title[which(grepl("Mlle.",titanic_train$Name_Title) | grepl("Ms.",titanic_train$Name_Title)
                               | grepl("Lady.",titanic_train$Name_Title)| grepl("Dona.",titanic_train$Name_Title))] <- "Miss."
titanic_test$Name_Title[which(grepl("Mlle.",titanic_test$Name_Title) | grepl("Ms.",titanic_test$Name_Title)
                               | grepl("Lady.",titanic_test$Name_Title)| grepl("Dona.",titanic_test$Name_Title))] <- "Miss."
titanic_train$Name_Title[titanic_train$Name_Title == 'Mme.']         <- 'Mrs.' 
titanic_test$Name_Title[titanic_test$Name_Title == 'Mme.']         <- 'Mrs.' 
# if not a big class --> put other
#- which is not a good practice, always avoid
titanic_train$Name_Title[-(which(grepl("Miss.",titanic_train$Name_Title) | grepl("Mr.",titanic_train$Name_Title) |
                                   grepl("Mrs.",titanic_train$Name_Title) |grepl("Master",titanic_train$Name_Title) ))] <- "Other"
titanic_test$Name_Title[-(which(grepl("Miss.",titanic_test$Name_Title) | grepl("Mr.",titanic_test$Name_Title) |
                                   grepl("Mrs.",titanic_test$Name_Title) |grepl("Master",titanic_test$Name_Title) ))] <- "Other"

# table(titanic_train['Name_Title']) #checking results
# table(titanic_test['Name_Title'])


# Creating a column with length of a name (more long --> more important--> more rich)
titanic_train['Name_Len'] = nchar(titanic_train$Name)
titanic_test['Name_Len'] = nchar(titanic_test$Name)
# cor(titanic_train$Name_Len,titanic_train$Fare)

# Adding Age-null flag
titanic_train['Is_age_null'] = 0
titanic_train$Is_age_null [which(is.na(titanic_train$Age) | (titanic_train$Age==""))] = 1
titanic_test['Is_age_null'] = 0
titanic_test$Is_age_null [which(is.na(titanic_test$Age) | (titanic_test$Age==""))] = 1

# Adding a flag for Children Age<18
titanic_train['Is_child'] = 0
titanic_train$Is_child [which(titanic_train$Age<18)] = 1
titanic_test['Is_child'] = 0
titanic_test$Is_child [which(titanic_test$Age<18)] = 1

# Creating information based on ticket column
titanic_train['Ticket_Len'] = nchar(titanic_train$Ticket)
titanic_test['Ticket_Len'] = nchar(titanic_test$Ticket)

# Code taken from Google --> applyign a function to identify the repeated tickets
##

full <- rbind(select(titanic_train,-c("Survived")), titanic_test)

ticket.unique <- rep(0, nrow(full))
tickets <- unique(full$Ticket)

for (i in 1:length(tickets)) {
  current.ticket <- tickets[i]
  party.indexes <- which(full$Ticket == current.ticket)
  for (k in 1:length(party.indexes)) {
    ticket.unique[party.indexes[k]] <- length(party.indexes)
  }
}

full$ticket.unique <- ticket.unique
full$ticket.size[full$ticket.unique == 1]   <- 'Single'
full$ticket.size[full$ticket.unique < 5 & full$ticket.unique>= 2]   <- 'Small'
full$ticket.size[full$ticket.unique >= 5]   <- 'Big'

titanic_train['Ticket_Size'] <-full[1:dim(titanic_train)[1], c("ticket.size")]
titanic_test['Ticket_Size'] <-full[(dim(titanic_train)[1]+1):(dim(titanic_train)[1]+dim(titanic_test)[1]), c("ticket.size")]
rm(full)

# take the first letter of the cabin
titanic_train['Cabin_Letter'] = substring(titanic_train$Cabin, 1,1)
titanic_test['Cabin_Letter']  = substring(titanic_test$Cabin, 1,1)
titanic_train$Cabin_Letter[which((titanic_train$Cabin_Letter=="")| (is.na(titanic_train$Cabin_Letter)))]<-"Unknown"
titanic_test$Cabin_Letter[which((titanic_test$Cabin_Letter=="")| (is.na(titanic_test$Cabin_Letter)))]<-"Unknown"

# Combine weak predictors to get a stronger predictor --> How big is my family
# Family type
titanic_train["Family_Type"] <-titanic_train$SibSp + titanic_train$Parch + 1 
titanic_train$Family_Type[titanic_train$Family_Type == 1] <- 'Single' 
titanic_train$Family_Type[titanic_train$Family_Type <= 4 & titanic_train$Family_Type >= 2] <- 'Small' 
titanic_train$Family_Type[titanic_train$Family_Type >= 5] <- 'Big' 

titanic_test["Family_Type"] <-titanic_test$SibSp + titanic_test$Parch + 1 
titanic_test$Family_Type[titanic_test$Family_Type == 1] <- 'Single' 
titanic_test$Family_Type[titanic_test$Family_Type <= 4 & titanic_test$Family_Type >= 2] <- 'Small' 
titanic_test$Family_Type[titanic_test$Family_Type >= 5] <- 'Big' 


############################################################     Removing unused columns and filling NA #############################################################
# To edit features --> restart from here

y_train = titanic_train$Survived
y_train <- y_train %>% as.factor
X_train = titanic_train

# Drop the Survived column
X_train<- select (X_train,-c("Survived"))
X_test = titanic_test


# Evaluating NA in columns colnames(titanic_train)
sum(is.na(titanic_train$Age)/sum(nrow(titanic_train)))    # 20%
sum(is.na(titanic_train$PClass)/sum(nrow(titanic_train))) #0% ok 
sum(is.na(titanic_train$Sex)/sum(nrow(titanic_train))) # 0% OK
sum(is.na(titanic_train$Cabin_Letter)/sum(nrow(titanic_train))) # 0% OK ....

#AGE: filling with simple mean
X_train$Age[is.na(X_train$Age)]<- mean(X_train$Age[!is.na(X_train$Age)])
#for test set--> still using the mean of train#filling with mode embarked
X_test$Age[is.na(X_test$Age)]<- mean(X_train$Age[!is.na(X_train$Age)])

#FARE: filling with mode --> it is a very skewed variable and it could alter the results
X_train$Fare[is.na(X_train$Fare)]<- statmod(X_train$Fare[!is.na(X_train$Fare)])
#for test set--> still using the mean of train#filling with mode embarked
X_test$Fare[is.na(X_test$Fare)]<- statmod(X_train$Fare[!is.na(X_train$Fare)])

#Embarked: filling with mode (is categorical)
X_train$Embarked <- replace(X_train$Embarked, which(is.na(X_train$Embarked)), statmod(X_train$Embarked))
X_test$Embarked <- replace(X_test$Embarked, which(is.na(X_test$Embarked)), statmod(X_train$Embarked))

############################################################     Preparing Data & Deleting columns #############################################################

#Deleting columns not useful --> this can be edited based on model performance
list_to_delete = c("PassengerId","Name","SibSp","Parch","Cabin","Ticket") #,"Embarked")

X_train = select(X_train,-list_to_delete)
X_test = select(X_test,-list_to_delete)

# converting to correct data types. First with factors
X_train$Family_Type <- X_train$Family_Type%>% as.factor
X_test$Family_Type <- X_test$Family_Type%>% as.factor
# not needed for Family type and Sex --> still good practice to be sure not to lose levels
X_train[,"Family_Type"] <- factor(X_train[,"Family_Type"], levels=levels(factor(c(levels(X_train$Family_Type),levels(X_test$Family_Type)))))
X_test[,"Family_Type"] <- factor(X_test[,"Family_Type"], levels=levels(factor(c(levels(X_train$Family_Type),levels(X_test$Family_Type)))))

X_train$Sex <- X_train$Sex%>% as.factor
X_test$Sex <- X_test$Sex%>% as.factor
X_train[,"Sex"] <- factor(X_train[,"Sex"], levels=levels(factor(c(levels(X_train$Sex),levels(X_test$Sex)))))
X_test[,"Sex"] <- factor(X_test[,"Sex"], levels=levels(factor(c(levels(X_train$Sex),levels(X_test$Sex)))))

#this loses the ordered aspect
X_train$Pclass<- X_train$Pclass %>% as.factor
X_test$Pclass<- X_test$Pclass %>% as.factor

X_train$Name_Title<- X_train$Name_Title %>% as.factor
X_test$Name_Title<- X_test$Name_Title %>% as.factor

X_train$Ticket_Size<- X_train$Ticket_Size %>% as.factor
X_test$Ticket_Size<- X_test$Ticket_Size %>% as.factor

X_train$Cabin_Letter<- X_train$Cabin_Letter %>% as.factor
X_test$Cabin_Letter<- X_test$Cabin_Letter %>% as.factor

X_train$Embarked<- X_train$Cabin_Letter %>% as.factor
X_test$Embarked<- X_test$Cabin_Letter %>% as.factor

# converting to numeric

X_train$Age<-X_train$Age %>% as.numeric
X_test$Age<-X_test$Age %>% as.numeric

X_train$Fare<-X_train$Fare %>% as.numeric
X_test$Fare<-X_test$Fare %>% as.numeric

X_train$Is_age_null<-X_train$Is_age_null %>% as.numeric
X_test$Is_age_null<-X_test$Is_age_null %>% as.numeric

X_train$Is_child<-X_train$Is_child %>% as.numeric
X_test$Is_child<-X_test$Is_child %>% as.numeric

X_train$Name_Len<-X_train$Name_Len %>% as.numeric
X_test$Name_Len <-X_test$Name_Len %>% as.numeric

X_train$Ticket_Len<-X_train$Name_Len %>% as.numeric
X_test$Ticket_Len <-X_test$Name_Len %>% as.numeric

# all is good? 
# str(X_train)
#Removing additional columns under need
#list_to_delete = c("Embarked")
#X_train = select(X_train,-list_to_delete)
#X_test = select(X_test,-list_to_delete)


################################################            Modeling    ################################################################

#######################################################
#################### Random Forest ####################
rf.1 <- randomForest(x=X_train, y=y_train, importance=TRUE, ntree=500)

rf.1
#Plotting Importance 
varImpPlot(rf.1)


set.seed(12) #per riproducibilitÃ  risultati

# k fold for estimating test error

n_folds <- 10 # k = 10
n_train<- nrow(X_train)
folds_i <- sample(rep(1:n_folds, length.out = n_train))
ntree<-seq(10,500,20)
fitted_models <- apply(t(ntree), 2, function(par) randomForest(x = X_train,y=y_train, ntree = par))

# we compute the train error using the confusion matrix
train_error <- sapply(fitted_models, function(obj) (obj$confusion[2]+obj$confusion[3])/nrow(X_train))
# let's compute the validation error
cv_tmp <- matrix(NA, nrow = n_folds, ncol = length(ntree))
for (k in 1:n_folds) {
  # here we select the validation set
  test_i <- which(folds_i == k) 
  train_xy <- X_train[-test_i, ]
  test_xy <- X_train[test_i, ]
  fitted_models <- apply(t(ntree), 2, function(par) randomForest(x = X_train,y=y_train, ntree = par))
  pred <- mapply(function(obj) predict(obj, data.frame(X_train)), fitted_models)
  
  cv_tmp[k, ] <- sapply(as.list(data.frame(pred)), function(y_hat) length(which(y_hat != y_train))/length(y_train))
  
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

bestRFmodel = fitted_models[[which.min(cv)]]

y_train_hat = predict(bestRFmodel,X_train) # predictions on original data


#Plotting ROC curve 
votes_1_RF <- bestRFmodel$votes[,1] # ranking of data points

pROC_obj <- roc(y_train,votes_1_RF,
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)

sens.ci <- ci.se(pROC_obj)
plot(sens.ci, type="shape", col="lightblue")

plot(sens.ci, type="bars")





#######################################################
####################### xgboost #######################

# cv:
n_folds <- 10 # k = 10
my_eta <- 0.32
my_max_depth <- 7
my_nthread <- 10
my_nrounds <- 2000

folds <- createFolds(y_train, k=n_folds)
# xgboost con cross validation 
xgbparam <- list(  objective           = "binary:logistic", 
                    booster             = "gbtree",
                    eval_metric         = "auc",
                    eta                 = my_eta,
                    max_depth           = my_max_depth,
                    subsample           = 0.8,
                    colsample_bytree    = 0.8,
                    nthread             = my_nthread
                    )
Xgbdtrain2 <- xgb.DMatrix(data=data.matrix(X_train),label=as.numeric(y_train)-1)
xgboostfit2 <- xgb.cv( params                   = xgbparam, 
                        data                     = Xgbdtrain2, 
                        nrounds                  = my_nrounds,
                        folds                    = folds,
                        early_stopping_rounds    = 50,
                        verbose                  = TRUE,
                        print_every_n            = 10L,
                        prediction               = TRUE,
                        maximize                 = FALSE
                        )
bestIteration <- xgboostfit2$best_iteration

xgboostfit2

# modello definitivo
model_trained <- xgb.train( params                    = xgbparam,
                            data                      = Xgbdtrain2,
                            watchlist                 = list(train=Xgbdtrain2),
                            nrounds                   = my_nrounds,
                            #early_stopping_rounds     = 1,
                            verbose                   = TRUE,
                            print_every_n             = 10L,
                            prediction                = TRUE
)

# setwd(currentWd)
# saveRDS(model_trained, "XGBoost_trained.rds")

xgb.names <- dimnames(data.matrix(X_train))[[2]]
importance_matrix <- xgb.importance(xgb.names, model = model_trained)
importance_matrix


Xgbdtest <- xgb.DMatrix(data=data.matrix(X_train), label=as.numeric(y_train)-1)
prediction <- predict(model_trained, Xgbdtest)
myRoc <- roc(as.numeric(y_train)-1, prediction)


cM <- confusionMatrix(as.factor(prediction), as.factor(y_train))



#Plotting ROC curve 
pROC_obj <- roc(y_train,prediction,
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)

sens.ci <- ci.se(pROC_obj)
plot(sens.ci, type="shape", col="lightblue")

plot(sens.ci, type="bars")

#### TO DO: fixing OVERFITTING IN XGBOOST







