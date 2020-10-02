whatAreFactors<-function(dataset){
  namesFactor<-c()
  for(i in 1:ncol(dataset)){
    if(is.factor(dataset[,i])) namesFactor<-c(namesFactor, colnames(dataset)[i])
  }
  return(namesFactor)
}