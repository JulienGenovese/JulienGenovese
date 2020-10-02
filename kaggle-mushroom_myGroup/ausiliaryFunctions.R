calculate_cramer <- function(m, DF) {
  for (r in seq(nrow(m))){
    for (c in seq(ncol(m))){
      m[[r, c]] <- assocstats(table(DF[[r]], DF[[c]]))$cramer
    }
  }
  return(m)
}
combine_column<-function(col1,col2){
  return(paste(col1,col2,sep=""))    
}