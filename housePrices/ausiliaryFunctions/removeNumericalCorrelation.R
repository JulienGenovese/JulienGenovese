removeNumericalCorrelation<-function(dataset){
  library(corrplot)
  correlations<-cor(subset(dataset, select=sapply(dataset, is.numeric)))
  corrplot(correlations, order = "hclust")
  highCorr<- findCorrelation(correlations, cutoff = .75)
  datasetNoCor<-dataset[,-highCorr]
  return(datasetNoCor)
}