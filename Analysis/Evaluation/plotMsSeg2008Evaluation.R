library( ANTsR )
library( reshape2 )
library( ggplot2 )

dataDirectory <- "/Users/ntustison/Data/WMH/MSSeg2008/"
source( "./getEvaluationMeasures.R" )

baseDirectory <- "./"
figuresDirectory <- paste0( baseDirectory, "../../Text/Figures/" )

measuresFile <- "../Data/measuresMsSeg2008.csv"
 
if( ! file.exists( measuresFile ) )
  {
    lesionFiles <- list.files( path = paste0( dataDirectory, "Nifti/" ), 
        pattern = "lesion.nii.gz", recursive = TRUE, full.names = TRUE )

    measuresDataFrame <- data.frame()
    for( i in seq.int( length( lesionFiles ) ) )
    {
    resultsDirectory <- paste0( sub( "Nifti", "Processed", dirname( lesionFiles[i] ) ), "/" )
    basePrefix <- sub( "lesion.nii.gz", "", basename( lesionFiles[i] ) )

    cat( "Processing ", basePrefix, "(", i, "out of", length( lesionFiles ), ")\n" )    

    reference <- antsImageRead( lesionFiles[i] )

    ewDavid <- antsImageRead( paste0( resultsDirectory, basePrefix, "ew_david_wmh.nii.gz" ) )
    ewDavid5 <- antsImageRead( paste0( resultsDirectory, basePrefix, "ew_david_wmh5.nii.gz" ) )
    ewDavidT1 <- antsImageRead( paste0( resultsDirectory, basePrefix, "ew_david_t1only_wmh.nii.gz" ) )
    ewDavid5T1 <- antsImageRead( paste0( resultsDirectory, basePrefix, "ew_david_t1only_wmh5.nii.gz" ) )
    sysu <- antsImageRead( paste0( resultsDirectory, basePrefix, "sysu_wmh.nii.gz" ) )

    segmentations <- list( ewDavid,
                            ewDavid5,
                            ewDavidT1,
                            ewDavid5T1,
                            sysu )
    
    segmentationNames <- c( "ewDavid", "ewDavid5", "ewDavidT1", "ewDavid5T1", "sysu" )
    
    pb <- txtProgressBar( min = 0, max = length( segmentations ), style = 3 )
    for( j in seq.int( length( segmentations ) ) )
        {
        setTxtProgressBar( pb, j )       
        measures <- getEvaluationMeasures( segmentations[[j]], reference )
        singleRow <- data.frame( Subject = basePrefix, Segmentation = segmentationNames[j], 
                                DiceOverlap = measures$DiceOverlap, 
                                HausdorffDistance = measures$HausdorffDistance,
                                HausdorffAverageDistance = measures$HausdorffAverageDistance, 
                                LogAbsoluteVolumetricDifference = measures$LogAbsoluteVolumetricDifference,
                                Recall = measures$Recall,
                                Precision = measures$Precision,
                                F1 = measures$F1 )
        if( nrow( measuresDataFrame ) == 0 )
        {
        measuresDataFrame <- singleRow    
        } else {
        measuresDataFrame <- rbind( measuresDataFrame, singleRow )
        }  
        write.csv( measuresDataFrame, measuresFile, row.names = FALSE, quote = FALSE )
        }
    cat( "\n" )  
    }
  }  

measuresDataFrame <- read.csv( measuresFile )
measuresDataFrame$Segmentation <- factor( measuresDataFrame$Segmentation )
measuresDataFrame$Subject <- NULL
for( i in 2:8 )
  {
  measuresDataFrame[,i] <- measuresDataFrame[,i] / max( measuresDataFrame[,i] )
  }

measuresMelt <- melt( measuresDataFrame, id.vars = c( "Segmentation" ),  
  measures.var = c( "DiceOverlap", "HausdorffDistance", "HausdorffAverageDistance", 
                    "LogAbsoluteVolumetricDifference", "Recall", "Precision", "F1" ),
  variable.name = "Measure" )
g <- ggplot( data = measuresMelt, aes( x = Measure, y = value, fill = Segmentation ) ) +
            geom_boxplot() + 
            ggtitle( "Normalized values" ) +
            theme( axis.text.x = element_text( angle = 45, vjust = 1, hjust = 1 ) )
plotFileName <- paste0( figuresDirectory, "MsSeg2008Evaluation.pdf" )
ggsave( plotFileName, g, width = 8, height = 5, units = "in" )
