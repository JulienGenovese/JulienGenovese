fillingByKNN <- function(dataset){
  dataset %<>% as_tibble() 
  namesNAColumns <- names(dataset)[which(colSums(is.na(dataset)) > 0)]
  #options(warn = 2)
  for(i in 1:length(namesNAColumns)){
    columnTofill <- namesNAColumns[i]
    print(columnTofill)
    columnToAvoid <- namesNAColumns[-(1:i)]
    
    df_x <- dataset %>% select(-c(columnTofill, columnToAvoid))
    df_y <- dataset[[columnTofill]]
    
    
    naEl <- is.na(df_y) # we split between the columns we want to fill
    x <- df_x[!naEl, ] # our learning set
    setToFill <- df_x[naEl, ] # our test set
    y <- df_y[!is.na(df_y)]
    
    dummies_modelFullRank <- dummyVars( ~ ., data = x , fullRank = TRUE)
    data_matFullRank <- predict(dummies_modelFullRank, newdata = x)
    nzv <- nearZeroVar(data_matFullRank)
    if(length(nzv)>0)
      data_matFullRank <- data_matFullRank[, -nzv]
    
    knnFit <- train(y = y,
                    x = data_matFullRank,
                    method = "knn",
                    trControl = trainControl(method = "repeatedcv", repeats = 5),
                    preProcess = c("center","scale"), 
                    tuneLength = 12
                    )
    #plot(knnFit)
    setToFill <- predict(dummies_modelFullRank, newdata = setToFill)
    if(length(nzv)>0)
      setToFill <- setToFill[, -nzv]
    if(class(setToFill) != "matrix"){
      setToFill <- matrix( setToFill, nrow = 1, ncol = length(setToFill))
      rownames(setToFill) <- "1"
      colnames(setToFill) <- colnames(data_matFullRank)
    }
    knnPredict <- predict(knnFit, newdata = setToFill)
    df_y[naEl] <- knnPredict
    dataset[, columnTofill] <- df_y
  }
  
  return(dataset)
}
