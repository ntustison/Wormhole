library( ggplot2 )

baseDirectory <- "./"
figuresDirectory <- paste0( baseDirectory, "../../Text/Figures/" )

wmhDataFrame <- read.csv( paste0( baseDirectory, "../Data/wmhOverlaps.csv" ) )
segmentationName <- c( "octant", "sliceWise7mb", "sliceWise300mb", "sliceWiseT1Only", "combined", "sysu",  "sysuAxial", "ucd" )
overlapMeasureName <- c( "TotalOverlap", "UnionOverlap", "MeanOverlap", "VolumeSimilarity", "FalseNegativeError", "FalsePositiveError" )  


for( m in seq.int( length( segmentationName ) ) )
  {
  if( m >= length( segmentationName ) )
    {
    break  
    }
  for( n in seq.int( m + 1, length( segmentationName ) ) )
    {
    whichPair <- paste0( segmentationName[m], ".", segmentationName[n] )

    pairwiseWmhDataFrame <- wmhDataFrame[which( wmhDataFrame$WhichPair == whichPair ),]  
    pairwiseWmhDataFrame$WhichOverlap <- factor( pairwiseWmhDataFrame$WhichOverlap, levels = overlapMeasureName )

    g <- ggplot( data = pairwiseWmhDataFrame, aes( x = WhichOverlap, y = OverlapMeasure, fill = WhichOverlap ) ) +
              geom_boxplot() +
              theme( axis.text.x = element_text( angle = 45, vjust = 1, hjust = 1 ) ) +
              ylim( -2, 2 ) +
              ggtitle( whichPair ) +
              theme( legend.position = "none" )
    plotFileName <- paste0( figuresDirectory, "Pairwise_", whichPair, ".pdf" )
    ggsave( plotFileName, g, width = 5, height = 5, units = "in" )

    }

  }
