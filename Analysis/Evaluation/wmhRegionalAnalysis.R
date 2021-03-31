library( ANTsR ) 
library( ANTsRNet )
library( randomForest )
library( cvms )
library( broom )    
library( ggimage )   
library( ggnewscale )
library( rsvg )    
library( pROC )

numberOfRuns <- 100
trainingPortion <- 0.8
verbose <- TRUE


covariates <- c( "Age_atFLAIR", "Gender", "Education", "BrainVolume", "Diagnosis" )
predictand <- c( "Trails" )
# region <- c( "total", "deep", "periventricular", "frontal", "parietal", "temporal", "occipital" )
region <- c( "total" )
isContinuous <- FALSE


################
#
# Organize data
#
################

baseDirectory <- "./"
figuresDirectory <- paste0( baseDirectory, "../../Text/Figures/" )

wmhDataFrame <- read.csv( paste0( baseDirectory, "../Data/wmhVolumes.csv" ) )
wmhDataFrame$Algorithm <- as.factor( wmhDataFrame$Algorithm )
algorithms <- levels( wmhDataFrame$Algorithm )

demoDataFrame <- read.csv( paste0( baseDirectory, "../Data/ADNI2_3_neuropsych_1423subj.csv" ) )

allDataFrame <- merge( wmhDataFrame, demoDataFrame, by.x = "Subject", by.y = "Image_ID" )

for( r in seq.int( region ) )
  {
  regionalColumns <- 6:13
  if( region[r] != "total" )  
    {
    regionalColumns <- grep( region[r], colnames( wmhDataFrame ) )
    }

  ################
  #
  # Run simulations
  #
  ################


  algorithmRmse <- array( data = 0, dim = c( numberOfRuns, length( algorithms ) ) )
  algorithmPredictions <- list()

  for( i in seq.int( length( algorithms ) ) )
    {
    cat( "Analyzing ", algorithms[i], "\n" )  
    if( verbose )
      {
      pb <- txtProgressBar( min = 1, max = numberOfRuns, style = 3 )
      }

    perAlgorithmDataFrame <- allDataFrame[allDataFrame$Algorithm == algorithms[i],]
    perAlgorithmDataFrame <- perAlgorithmDataFrame[complete.cases( perAlgorithmDataFrame ),]
    perAlgorithmDataFrame$WmhVolume <- rowSums( perAlgorithmDataFrame[,regionalColumns])
    if( ! isContinuous )
      {
      predictandColumn <- which( colnames( perAlgorithmDataFrame ) == predictand[1] )
      perAlgorithmDataFrame[, predictandColumn] <- as.factor( perAlgorithmDataFrame[, predictandColumn] )
      }


    numberOfSubjects <- nrow( perAlgorithmDataFrame )
    
    wmhFormula <- as.formula( paste0( predictand[1], "~ WmhVolume + ", paste( covariates, collapse = "+" ) ) ) 

    for( j in seq.int( numberOfRuns ) )
      {
      setTxtProgressBar( pb, j )  
      trainingIndices <- sample.int( numberOfSubjects, size = floor( trainingPortion * numberOfSubjects ) )
        
      trainingData <- perAlgorithmDataFrame[trainingIndices,]
      testingData <- perAlgorithmDataFrame[-trainingIndices,]
      
      if( isContinuous )
        {
        rf <- randomForest( wmhFormula, data = trainingData, importance = TRUE )

        predictions <- predict( rf, newdata = testingData )

        N <- length( predictions )
        rmse <- sqrt( sum( ( testingData$Age_atFLAIR - predictions )^2 ) / N )
        algorithmRmse[j, i] <- rmse
        } else {
        rf <- randomForest( wmhFormula, data = trainingData, importance = TRUE )

        predictions <- as.data.frame( predict( rf, newdata = testingData, type = "prob" ) )
        predictions$predict <- factor( colnames( predictions )[1:2][apply( predictions[,1:2], 1, which.max )] )
        predictions$observed <- perAlgorithmDataFrame[-trainingIndices, predictandColumn]

        if( j == 1 )
          {
          algorithmPredictions[[i]] <- predictions
          } else {
          algorithmPredictions[[i]] <- rbind( algorithmPredictions[[i]], predictions )
          }
        }
      }
    cat( "\n" )  
    }  

  if( isContinuous )
    {
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
            ggtitle( region[r] )

    plotFileName <- paste0( figuresDirectory, "predict_", predictand[1], "_", region[r], ".pdf" )
    ggsave( plotFileName, g, width = 8, height = 5, units = "in" )    
    cat( "\n", colnames( algorithmRmse ), "\n", colMeans( algorithmRmse ), "\n\n" )
    } else {
    roc.dx <- list()
    for( p in seq.int( length( algorithmPredictions ) ) )
      {
      roc.dx[[p]] <- roc( algorithmPredictions[[p]]$observed, as.numeric( algorithmPredictions[[p]][,1] ) )
      cat( algorithms[p], ": AUC = ", roc.dx[[p]]$auc, "\n", sep = "" )
      }

    # algorithms <- c( "Sysu", "SysuAxial", "UCD", "Octant", "Slicewise7mb", "Slicewise300mb", "SlicewiseT1Only", "Combined" )
    
    g <- ggroc( list( Combined = roc.dx[[1]], Octant = roc.dx[[2]], SW300 = roc.dx[[3]], SW7 = roc.dx[[4]], 
                      SWT1 = roc.dx[[5]], Sysu = roc.dx[[6]], SysuAxial = roc.dx[[7]], UCD = roc.dx[[8]] ), size = 0.5, legacy.axes = "TRUE" ) +
      geom_abline( intercept = 0, slope = 1, color = "darkgrey", linetype = "dashed" ) +
      labs( color = "Pipeline" ) +
      theme( legend.position = "bottom" ) +
      theme( legend.title = element_blank() )
    plotFileName <- paste0( figuresDirectory, "predict_", predictand[1], "_", region[r], ".pdf" )
    ggsave( plotFileName, g, width = 5, height = 5, units = "in" )    
    }

  }