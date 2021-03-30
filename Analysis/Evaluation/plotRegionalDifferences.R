library( ANTsR ) 
library( ANTsRNet )
library( ggplot2 )


region <- c( "total", "deep", "periventricular", "frontal", "parietal", "temporal", "occipital" )
doRelative <- FALSE

baseDirectory <- "./"
figuresDirectory <- paste0( baseDirectory, "../../Text/Figures/" )

wmhDataFrame <- read.csv( paste0( baseDirectory, "../Data/wmhVolumes.csv" ) )
wmhDataFrame$Algorithm <- as.factor( wmhDataFrame$Algorithm )
algorithms <- levels( wmhDataFrame$Algorithm )


for( r in seq.int( region ) )
  {
  regionalColumns <- 6:13
  if( region[r] != "total" )  
    {
    regionalColumns <- grep( region[r], colnames( wmhDataFrame ) )
    }

  wmhRegionalVolumes <- list()
  for( i in seq.int( length( algorithms ) ) )
    {
    perAlgorithmDataFrame <- wmhDataFrame[wmhDataFrame$Algorithm == algorithms[i],]
    perAlgorithmDataFrame <- perAlgorithmDataFrame[complete.cases( perAlgorithmDataFrame ),]

    wmhRegionalVolumes[[i]] <- rowSums( perAlgorithmDataFrame[, regionalColumns] )
    }  
  
  if( doRelative == TRUE )
    {
    wmhs <- list2DF( wmhRegionalVolumes )  
    wmhs <- t( scale( t( wmhs ), center = TRUE, scale = TRUE ) )
    colnames( wmhs ) <- algorithms
    for( i in seq.int( length( algorithms ) ) )
      {
      wmhRegionalVolumes[[i]] <- wmhs[,i]
      }
    }

  wmhPlotDataFrame <- data.frame( Algorithm = character(), WMH = double() )
  for( i in seq.int( length( algorithms ) ) )
    {
    if( i == 1 )  
      {
      wmhPlotDataFrame <- data.frame( Algorithm = rep( algorithms[i], length( wmhRegionalVolumes[[i]] ) ),
                                      WMH = wmhRegionalVolumes[[i]] )  
      } else {
      wmhPlotDataFrame <- rbind( wmhPlotDataFrame, 
                                 data.frame( Algorithm = rep( algorithms[i], length( wmhRegionalVolumes[[i]] ) ),
                                      WMH = wmhRegionalVolumes[[i]] ) )
      }
    }

  g <- ggplot( data = wmhPlotDataFrame, aes( x = Algorithm, y = WMH, fill = Algorithm ) ) +
            geom_boxplot() +
            theme( axis.text.x = element_text( angle = 45, vjust = 1, hjust = 1 ) ) +
            ggtitle( region[r] )
  plotFileName <- paste0( figuresDirectory, "RegionalVolumes_", region[r], ".pdf" )
  if( doRelative == TRUE )            
    {
    plotFileName <- paste0( figuresDirectory, "RegionalRelativeVolumes_", region[r], ".pdf" )  
    }
  ggsave( plotFileName, g, width = 8, height = 5, units = "in" )
  }