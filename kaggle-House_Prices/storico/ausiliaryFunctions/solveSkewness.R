solveSkewness<-function(dataset){
  datasetTrans<-dataset
  for(i in 1:ncol(datasetTrans)){
    if(class(datasetTrans[,i])!="factor"){
      responseSkew <- BoxCoxTrans(dataset[,i])
      datasetTrans[,i]<-predict(responseSkew, dataset[,i])
    }
  }
  return(datasetTrans)
}