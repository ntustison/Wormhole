library( ANTsRNet )
library( ANTsR ) 

baseDirectory <- "/Users/ntustison/Data/WMH/ADNI/ADNI_temp"

octants <- list.files( path = baseDirectory, pattern = "EwDavidOctantWmhSegmentation.nii.gz", 
  recursive = TRUE, full.names = TRUE )

wmhDataFrame <- data.frame()

segmentationName <- c( "octant", "sliceWise7mb", "sliceWise300mb", "sliceWiseT1Only", "combined", "sysu",  "sysuAxial", "ucd" )
overlapMeasureName <- c( "TotalOverlap", "UnionOverlap", "MeanOverlap", "VolumeSimilarity", "FalseNegativeError", "FalsePositiveError" )  

pb <- txtProgressBar( min = 0, max = length( octants ), style = 3 )
# for( i in seq.int( 5 ) )  
for( i in seq.int( length( octants ) ) )  
  {
  tokens <- strsplit( octants[i], "/" )   
  subjectId <- tokens[[1]][8]
  date <- tokens[[1]][10]
  seriesId <- tokens[[1]][11]

  setTxtProgressBar( pb, i )  
  brainSegT1 <- antsImageRead( sub( "FLAIR_EwDavidOctantWmhSegmentation", "BrainSegmentation", octants[i] ) )
  t1xflairXfrm <- sub( "FLAIR_EwDavidOctantWmhSegmentation.nii.gz", "T1xFLAIR0GenericAffine.mat", octants[i] )  

  octant <- antsImageRead( octants[i] )
  sliceWise7mb <- antsImageRead( sub( "EwDavidOctantWmhSegmentation", "EwDavidSlicewiseWmhSegmentation_7mb", octants[i] ) )
  sliceWise300mb <- antsImageRead( sub( "EwDavidOctantWmhSegmentation", "EwDavidSlicewiseWmhSegmentation_300mb", octants[i] ) )
  sliceWiseT1Only <- antsImageRead( sub( "EwDavidOctantWmhSegmentation", "EwDavidSlicewiseWmhSegmentation_T1Only", octants[i] ) )
  combined <- thresholdImage( antsCopyImageInfo( sliceWise300mb, octant ) + sliceWise300mb, 0, 0, 0, 1 )

  sysu <- antsImageRead( sub( "EwDavidOctantWmhSegmentation", "SysuWmhSegmentation", octants[i] ) )
  sysuAxial <- antsImageRead( sub( "EwDavidOctantWmhSegmentation", "SysuWmhAxialSegmentation", octants[i] ) )
  
  ucd <- thresholdImage( antsImageRead( sub( "ants_FLAIR_EwDavidOctantWmhSegmentation", "itk_FLAIR_WMH_Native", octants[i] ) ), 0, 0, 0, 1 )
  
  whiteMatterMask <- thresholdImage( brainSegT1, 3, 3, 1, 0 )
  whiteMatterMask <- antsApplyTransforms( octant, whiteMatterMask, 
    transformlist = c( t1xflairXfrm ), whichtoinvert = c( TRUE ),
    interpolator = "nearestNeighbor" )

  volumePerVoxel <- prod( antsGetSpacing( whiteMatterMask ) )
  brainVolume <- sum( brainSegT1 ) * prod( antsGetSpacing( brainSegT1 ) )

  segmentations <- list( octant, 
                         sliceWise7mb,
                         sliceWise300mb,
                         sliceWiseT1Only,
                         combined,
                         sysu, 
                         sysuAxial,
                         ucd )

  overlapNames <- c()
  for( m in seq.int( length( segmentations ) ) )
    {
    source <- segmentations[[m]] * antsCopyImageInfo( segmentations[[m]], whiteMatterMask )
    if( m >= length( segmentations ) )
      {
      break  
      }
    for( n in seq.int( m + 1, length( segmentations ) ) )
      {
      target <- segmentations[[n]] * antsCopyImageInfo( segmentations[[n]], whiteMatterMask )
      overlap <- labelOverlapMeasures( source, antsCopyImageInfo( source, target ) )
      
      whichPair <- paste0( segmentationName[m], ".", segmentationName[n] )

      for( j in seq.int( length( overlapMeasureName ) ) )
        {
        singleRow <- c( subjectId, date, seriesId, brainVolume, whichPair, overlapMeasureName[j], overlap[1,j+1] )  
        if( nrow( wmhDataFrame ) == 0 )
          {
          wmhDataFrame <- t( as.data.frame( singleRow ) )
          } else {
          wmhDataFrame <- rbind( wmhDataFrame, singleRow )
          }
        }  
      }  
    }                       
  }
cat( "\n" )  

colnames( wmhDataFrame ) <- c( "Subject", "Date", "Series", "BrainVolume", "WhichPair", "WhichOverlap", "OverlapMeasure" )
rownames( wmhDataFrame ) <- NULL

write.csv( wmhDataFrame, file = "../Data/wmhOverlaps.csv", quote = FALSE, row.names = FALSE )