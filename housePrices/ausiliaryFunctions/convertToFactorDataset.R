convertToFactorDataset<-function(dataset, namesFactor){
  for (i in 1:ncol(dataset)){
    toMatch<-colnames(dataset)[i] 
    for(j in 1:length(namesFactor)){
      if(grepl(namesFactor[j],toMatch)){
        dataset[,i]<-as.factor(dataset[,i])
        break
      } 
    }
  }
  return(dataset)
}