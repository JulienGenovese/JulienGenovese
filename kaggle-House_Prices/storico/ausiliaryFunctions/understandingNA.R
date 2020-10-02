understandingNA <- function(dataset){

  dataset$Alley[is.na(dataset$Alley)] <- "No alley access"
  dataset$BsmtQual[is.na(dataset$BsmtQual)] <- "No basement"
  dataset$BsmtCond[is.na(dataset$BsmtCond)] <- "No basement"
  dataset$BsmtExposure[is.na(dataset$BsmtExposure)] <- "No basement"
  dataset$BsmtFinType1[is.na(dataset$BsmtFinType1)] <- "No basement"
  dataset$BsmtFinType2[is.na(dataset$BsmtFinType2)] <- "No basement"
  dataset$FireplaceQu[is.na(dataset$FireplaceQu)] <- "No fireplace"
  dataset$GarageType[is.na(dataset$GarageType)] <- "No garage"
  dataset$GarageFinish[is.na(dataset$GarageFinish)] <- "No garage"
  dataset$GarageQual[is.na(dataset$GarageQual)] <- "No garage"
  dataset$GarageCond[is.na(dataset$GarageCond)] <- "No garage"
  dataset$PoolQC[is.na(dataset$PoolQC)] <- "No pool"
  dataset$Fence[is.na(dataset$Fence)] <- "No fence"
  dataset$MiscFeature[is.na(dataset$MiscFeature)] <- "None"
  return(dataset)
}