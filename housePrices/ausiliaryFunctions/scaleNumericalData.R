scaleNumericalData<-function(dataset){
  for(i in 1 : ncol(dataset)){
    if(is.numeric(dataset[,i])){
      mean<-mean(dataset[,i])
      std<-sd(dataset[,i])
      dataset[,i]<-(dataset[,i]-mean)/std
    }
  }
  return(dataset)
}