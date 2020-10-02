percentageNumCol<-function(dataset, typeofAnalysis = 1, numCol = 0){
  num<-0
  for(i in 1:ncol(dataset)){
    if(is.numeric(dataset[,i])) num = num+1
  }
  if(typeofAnalysis==1){
    print(paste0("The percentage of numerical columns is ",round(100*num/ncol(dataset)),"%"))
  }else{
    print(paste0("The percentage of numerical columns relative to initial dataset is ",round(100*num/numCol),"%"))
  }
}