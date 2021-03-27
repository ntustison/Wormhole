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
doBinaryClassification <- TRUE

algorithms <- c( "Sysu", "SysuAxial", "UCD", "Octant", "Slicewise7mb", "Slicewise300mb", "SlicewiseT1Only", "Combined" )
palettes <- c( "Greens", "Greens", "Oranges", "Purples", "Purples", "Purples", "Reds", "Purples" )
covariates <- c( "Gender", "Age_atFLAIR", "Education", "BrainVolume" )

################
#
# Organize data
#
################

baseDirectory <- "/Users/ntustison/Data/WMH/"

wmhDataFrame <- read.csv( paste0( baseDirectory, "/Scripts/wmhVolumes.csv" ) )
mygg <- ggpairs( wmhDataFrame, columns = 5:12 )
ggsave( "~/Desktop/wmhs.pdf", mygg )

demoDataFrame <- read.csv( paste0( baseDirectory, "/Scripts/ADNI2_3_neuropsych_1423subj.csv" ) )
allDataFrame <- merge( wmhDataFrame, demoDataFrame, by.x = "Subject", by.y = "Image_ID" )
allDataFrame <- allDataFrame[complete.cases(allDataFrame),]

################
#
# Run simulations
#
################

if( doBinaryClassification == FALSE )
  {
  allDataFrame$Diagnosis <- factor( allDataFrame$Diagnosis, levels = c( "CN", "MCI", "Dementia" ) )

  numberOfSubjects <- nrow( allDataFrame )

  for( i in seq.int( length( algorithms ) ) )
    {
    cat( "Analyzing ", algorithms[i], "\n" )  
    if( verbose )
      {
      pb <- txtProgressBar( min = 1, max = numberOfRuns, style = 3 )
      }
    dxFormula <- as.formula( paste0( "Diagnosis ~ ", algorithms[i], " + ", paste( covariates, collapse = "+" ) ) ) 

    algorithmPredictions <- data.frame()
    for( j in seq.int( numberOfRuns ) )
      {
      setTxtProgressBar( pb, j )  
      trainingIndices <- sample.int( numberOfSubjects, size = floor( trainingPortion * numberOfSubjects ) )
        
      trainingData <- allDataFrame[trainingIndices,]
      testingData <- allDataFrame[-trainingIndices,]
      
      rf <- randomForest( dxFormula, data = trainingData, importance = TRUE )

      predictions <- as.data.frame( predict( rf, newdata = testingData, type = "prob" ) )
      predictions$predict <- factor( colnames( predictions )[1:3][apply( predictions[,1:3], 1, which.max )], levels = c( "CN", "MCI", "Dementia" ) )
      predictions$observed <- allDataFrame$Diagnosis[-trainingIndices]

      if( j == 1 )
        {
        algorithmPredictions <- predictions
        } else {
        algorithmPredictions <- rbind( algorithmPredictions, predictions )
        }
      }

    algorithmConfusionMatrix <- confusion_matrix( targets = algorithmPredictions$observed,
                                                  predictions = algorithmPredictions$predict )
    
    confusionPlot <- plot_confusion_matrix(
        algorithmConfusionMatrix$`Confusion Matrix`[[1]],
        add_sums = TRUE,
        class_order = c( "CN", "MCI", "Dementia" ),
        palette = palettes[i],
        sums_settings = sum_tile_settings(
            palette = "Greys",
            label = "Total",
            tc_tile_border_color = "black"
            )
        )           
    ggsave( paste0( "~/Desktop/wmhDxConfusionMatrix_", algorithms[i], ".pdf" ), confusionPlot )    

    cat( "\n" )  
    }  
  } else {
  allDataFrame <- allDataFrame[-which( allDataFrame$Diagnosis == "MCI" ),]  
  allDataFrame$Diagnosis <- factor( allDataFrame$Diagnosis, levels = c( "CN", "Dementia" ) )

  numberOfSubjects <- nrow( allDataFrame )


  algorithmPredictions <- list()
  for( i in seq.int( length( algorithms ) ) )
    {
    cat( "Analyzing ", algorithms[i], "\n" )  
    if( verbose )
      {
      pb <- txtProgressBar( min = 1, max = numberOfRuns, style = 3 )
      }
    dxFormula <- as.formula( paste0( "Diagnosis ~ ", algorithms[i], " + ", paste( covariates, collapse = "+" ) ) ) 

    algorithmPredictions[[i]] <- data.frame()
    for( j in seq.int( numberOfRuns ) )
      {
      setTxtProgressBar( pb, j )  
      trainingIndices <- sample.int( numberOfSubjects, size = floor( trainingPortion * numberOfSubjects ) )
        
      trainingData <- allDataFrame[trainingIndices,]
      testingData <- allDataFrame[-trainingIndices,]
      
      rf <- randomForest( dxFormula, data = trainingData, importance = TRUE )

      predictions <- as.data.frame( predict( rf, newdata = testingData, type = "prob" ) )
      predictions$predict <- factor( colnames( predictions )[1:2][apply( predictions[,1:2], 1, which.max )], levels = c( "CN", "Dementia" ) )
      predictions$observed <- allDataFrame$Diagnosis[-trainingIndices]

      if( j == 1 )
        {
        algorithmPredictions[[i]] <- predictions
        } else {
        algorithmPredictions[[i]] <- rbind( algorithmPredictions[[i]], predictions )
        }
      }
    cat( "\n" )  
    }  

  roc.dx <- list()
  for( p in seq.int( length( algorithmPredictions ) ) )
    {
    roc.dx[[p]] <- roc( algorithmPredictions[[p]]$observed, as.numeric( algorithmPredictions[[p]][,1] ) )
    cat( algorithms[p], ": AUC = ", roc.dx[[p]]$auc, "\n", sep = "" )
    }

  # algorithms <- c( "Sysu", "SysuAxial", "UCD", "Octant", "Slicewise7mb", "Slicewise300mb", "SlicewiseT1Only", "Combined" )
  
  g <- ggroc( list( Sysu = roc.dx[[1]], SysuAxial = roc.dx[[2]], UCD = roc.dx[[3]], Octant = roc.dx[[4]], 
                    SW7 = roc.dx[[5]], SW300 = roc.dx[[6]], SWT1 = roc.dx[[7]], Combined = roc.dx[[8]] ), size = 0.5, legacy.axes = "TRUE" ) +
    geom_abline( intercept = 0, slope = 1, color = "darkgrey", linetype = "dashed" ) +
    labs( color = "Pipeline" ) +
    ggtitle( "CN vs. Dementia" ) +
    theme( legend.position = "bottom" ) +
    theme( legend.title = element_blank() )

  ggsave( paste0( "~/Desktop/wmhDxRoc.pdf" ), g, width = 5, height = 5, units = "in" )

  }
