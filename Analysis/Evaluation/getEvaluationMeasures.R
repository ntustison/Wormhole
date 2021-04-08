library( ANTsR )

getEvaluationMeasures <- function( sourceImage, targetImage )
{

  overlap <- labelOverlapMeasures( sourceImage, targetImage )

  # Check if multi-label

  if( length( overlap$MeanOverlap ) > 2 )
    {
    stop( "Source or target image has more than one label." )
    }

  #################
  #
  # Get dice metric
  #
  #################

  dice <- NA
  if( length( overlap$MeanOverlap ) > 1 )
    {
    dice <- overlap$MeanOverlap[2]
    }

  #################
  #
  # Get hausdorff distance
  #
  #################

  hausdorff <- hausdorffDistance( sourceImage, targetImage )

  #################
  #
  # Get absolute log-transformed volume difference
  #
  #################

  logAvd <- abs( log( sum( sourceImage ) / sum( targetImage ) ) )

  #################
  #
  # Get recall, precision, and F1
  #
  #################

  ccSource <- labelClusters( sourceImage, minClusterSize = 0, fullyConnected = TRUE )
  ccTarget <- labelClusters( targetImage, minClusterSize = 0, fullyConnected = TRUE )

  # subtract 1 for the background

  numWmhIntersection <- length( unique( sourceImage * ccTarget ) ) - 1
  numWmhTarget <- length( unique( ccTarget ) ) - 1

  recall <- 1.0
  if( numWmhTarget > 0 )
    {
    recall <- numWmhIntersection / numWmhTarget
    }

  numWmhIntersection <- length( unique( ccSource * targetImage ) ) - 1
  numWmhSource <- length( unique( ccSource ) ) - 1

  precision <- 1.0
  if( numWmhTarget > 0 )
    {
    precision <- numWmhIntersection / numWmhSource
    }

  f1 <- 0
  if( precision + recall > 0.0 )
    {
    f1 <- 2.0 * ( precision * recall ) / ( precision + recall )
    }

  return( list( DiceOverlap = dice,
                HausdorffDistance = hausdorff$Distance,
                HausdorffAverageDistance = hausdorff$AverageDistance,
                LogAbsoluteVolumetricDifference = logAvd,
                Recall = recall,
                Precision = precision,
                F1 = f1 ) )
}