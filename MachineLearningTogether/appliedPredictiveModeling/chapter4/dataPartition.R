library(caret)

data(oil)
str(oilType)
table(oilType)

trainOil<-sample(oilType,60)
table(trainOil)

soStr<-createDataPartition(oilType, p = 0.625, list = FALSE)
trainOil2<-oilType[soStr]
table(trainOil2)

binom.test(26,30)
