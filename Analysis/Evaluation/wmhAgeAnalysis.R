library( ANTsR ) 
library( ANTsRNet )
library( randomForest )
library( cvms )
library( broom )    
library( ggimage )   
library( ggnewscale )
library( rsvg )    


numberOfRuns <- 100
trainingPortion <- 0.8
verbose <- TRUE

algorithms <- c( "Sysu", "SysuAxial", "UCD", "Octant", "Slicewise7mb", "Slicewise300mb", "SlicewiseT1Only", "Combined" )
covariates <- c( "Gender", "Education", "BrainVolume", "Diagnosis" )

################
#
# Organize data
#
################

baseDirectory <- "/Users/ntustison/Data/WMH/"

wmhDataFrame <- read.csv( paste0( baseDirectory, "/Scripts/wmhVolumes.csv" ) )
demoDataFrame <- read.csv( paste0( baseDirectory, "/Scripts/ADNI2_3_neuropsych_1423subj.csv" ) )
allDataFrame <- merge( wmhDataFrame, demoDataFrame, by.x = "Subject", by.y = "Image_ID" )
allDataFrame <- allDataFrame[complete.cases(allDataFrame),]

################
#
# Run simulations
#
################


numberOfSubjects <- nrow( allDataFrame )

algorithmRmse <- array( data = 0, dim = c( numberOfRuns, length( algorithms ) ) )
for( i in seq.int( length( algorithms ) ) )
  {
  cat( "Analyzing ", algorithms[i], "\n" )  
  if( verbose )
    {
    pb <- txtProgressBar( min = 1, max = numberOfRuns, style = 3 )
    }
  ageFormula <- as.formula( paste0( "Age_atFLAIR ~ ", algorithms[i], " + ", paste( covariates, collapse = "+" ) ) ) 

  for( j in seq.int( numberOfRuns ) )
    {
    setTxtProgressBar( pb, j )  
    trainingIndices <- sample.int( numberOfSubjects, size = floor( trainingPortion * numberOfSubjects ) )
      
    trainingData <- allDataFrame[trainingIndices,]
    testingData <- allDataFrame[-trainingIndices,]
    
    rf <- randomForest( ageFormula, data = trainingData, importance = TRUE )

    predictions <- predict( rf, newdata = testingData )

    N <- length( predictions )
    rmse <- sqrt( sum( ( testingData$Age_atFLAIR - predictions )^2 ) / N )
    algorithmRmse[j, i] <- rmse
    }
  cat( "\n" )  
  }  

# algorithms <- c( "Sysu", "SysuAxial", "UCD", "Octant", "Slicewise7mb", "Slicewise300mb", "SlicewiseT1Only", "Combined" )

colnames( algorithmRmse ) <- algorithms

algorithmRmseDataFrame <- data.frame( Algorithm = character(), RMSE = numeric() )
for( i in seq.int( length( algorithms ) ) )
  {
  if( i == 1 )  
    {
    algorithmRmseDataFrame <- data.frame( Algorithm = rep( algorithms[i], numberOfRuns ),
                                          RMSE = algorithmRmse[,i] )
    } else {
    algorithmRmseDataFrame <- rbind( algorithmRmseDataFrame,
                                data.frame( Algorithm = rep( algorithms[i], numberOfRuns ),
                                            RMSE = algorithmRmse[,i] ) )
    }
  }

g <- ggplot( data = algorithmRmseDataFrame, aes( RMSE, fill = Algorithm, colour = Algorithm ) ) +
        geom_density( alpha = 0.1 ) +
        ggtitle( "Age prediction")
ggsave( paste0( "~/Desktop/wmhAgeRmse.pdf" ), g, width = 8, height = 5, units = "in" )
colMeans( algorithmRmse )