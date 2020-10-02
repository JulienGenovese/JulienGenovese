dealingWithNA<-function(dataset, method, coltrainset, namesFactor){

  #  variance near zero: I delete them
  nzv <- nearZeroVar(dataset, saveMetrics = TRUE)

  # this is the case in which we have very low variance and possibles NA values that we 
  # can't treat with knn 
  datasetNoVar<- dataset[, !nzv[,"nzv"]]

  convertToFactorDataset(datasetNoVar,namesFactor) %>% percentageNumCol(.,2,ncol(trainset))

  # we have reduce the number of columns for this reason:
  # we don't want predictor with low variability and NA.
  # But we will recover predictors with low variability but without N
  
  
  # Computation of NA percentage
  par(mfrow=c(1,2))
  naPercRow<-rowSums(is.na(datasetNoVar))/ncol(datasetNoVar)
  hist(naPercRow) # We don't see a row with a lot of NA (see values)
  naPercCol<-colSums(is.na(datasetNoVar))/nrow(datasetNoVar)
  hist(naPercCol) # we have some columns with only NA (see values)
  threshold<-0.5 # over this percentage we remove the column
  enoughdata<-which(naPercCol<threshold)
  datasetNoVar<-datasetNoVar[,enoughdata] # i remove definitively the columns with too many NA
  
  naPercCol<-colSums(is.na(datasetNoVar))/nrow(datasetNoVar) # i compute again what are the interesting columns
  naPercCol<-sort(naPercCol[naPercCol>0])
  namesNAColumns<-names(naPercCol)
  
  # filling methods 
  if (method == "julien"){
    datasetNoVarRegFilled<-fillingBySimpleKNN(datasetNoVar,namesNAColumns, namesFactor)
    
  }else if(method =="valerio"){
    next # 
  }
  
  datasetNoVarRegFilled %>% percentageNumCol(.,2,coltrainset)
  # we recover predictor with near zero variance but with possible NA
  dataset <- cbind(datasetNoVarRegFilled %>% select(namesNAColumns),
                   dataset %>% select(-namesNAColumns))
  # we remove the ones that have NA 
  resNA<- colSums(is.na(dataset))
  resNA<- names(resNA[resNA>0])
  dataset<-dataset %>% select(-resNA)
  dataset<-convertToFactorDataset(dataset, namesFactor)
  dataset %>% percentageNumCol(., 2, coltrainset)
  return(dataset)
}