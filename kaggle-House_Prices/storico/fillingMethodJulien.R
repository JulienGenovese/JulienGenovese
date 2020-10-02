fillingBySimpleKNN<-function(dataset,namesNAColumns, namesFactor){
  
  set.seed(400)
  dataset<-convertToFactorDataset(dataset, namesFactor)
  options(warn = 2)
  for (i in 1:length(namesNAColumns)){
    columnTofill<-namesNAColumns[i]
    print(columnTofill)
    columnToAvoid<-namesNAColumns[-(1:i)]
    
    df_x<-dataset %>% select(-c(columnTofill,columnToAvoid))
    df_y<-dataset[[columnTofill]]
    
    nzv <- nearZeroVar(df_x)
    if(length(nzv)>0) df_x <- df_x[, -nzv]
    naEl<-is.na(df_y) # we split between the column we want to fill
    x<-df_x[!naEl,] # our learning set
    setToFill<-df_x[naEl,] # our test set
    y<-df_y[!is.na(df_y)]
    
    knnFit <- train(x = x,y = y,  method = "knn", trControl = trainControl(method="repeatedcv",repeats = 3), 
                  preProcess = c("center","scale"), 
                  tuneLength = 10)
    plot(knnFit)
    knnPredict <- predict(knnFit,newdata = setToFill)
    df_y[naEl]<-knnPredict
    dataset[,columnTofill]<-df_y
  }
  
  return(dataset)
}