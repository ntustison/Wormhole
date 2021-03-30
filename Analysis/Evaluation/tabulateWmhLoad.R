library( ANTsRNet )
library( ANTsR ) 

baseDirectory <- "/Users/ntustison/Data/WMH/ADNI/ADNI_temp"

octants <- list.files( path = baseDirectory, pattern = "EwDavidOctantWmhSegmentation.nii.gz", 
  recursive = TRUE, full.names = TRUE )

wmhDataFrame <- data.frame()

regions <- as.vector( outer( c( "frontal", "parietal", "temporal", "occipital" ), 
  c( "periventricular", "deep" ), paste, sep = "." ) )
segmentationNames <- c( "octant", "sliceWise7mb", "sliceWise300mb", "sliceWiseT1Only", "combined", "sysu",  "sysuAxial", "ucd" )  

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

  dktFile <- sub( "FLAIR_EwDavidOctantWmhSegmentation", "dkt", octants[i] )
  dktLobesFile <- sub( "FLAIR_EwDavidOctantWmhSegmentation", "dktLobes", octants[i] )
  if( ! file.exists( dktFile ) || ! file.exists( dktLobesFile ) )
    {
    next  
    }

  dkt <- antsImageRead( dktFile )
  dktLobes <- antsImageRead( dktLobesFile )
  
  distance <- antsImageClone( dkt ) * 0
  distance[dkt == 4 | dkt == 43] <- 1
  distance <- iMath( distance, "MaurerDistance" )

  # t1Combined <- antsImageRead( sub( "FLAIR_EwDavidOctantWmhSegmentation", "T1_combinedWMH", octants[i] ) )
  # t1Combined <- antsApplyTransforms( octant, t1Combined, 
  #   transformlist = c( t1xflairXfrm ), whichtoinvert = c( TRUE ),
  #   interpolator = "nearestNeighbor" )
  
  whiteMatterMask <- thresholdImage( brainSegT1, 3, 3, 1, 0 ) * dktLobes
  whiteMatterMask[whiteMatterMask == 1 & distance > 10] <- 5  
  whiteMatterMask[whiteMatterMask == 2 & distance > 10] <- 6
  whiteMatterMask[whiteMatterMask == 3 & distance > 10] <- 7
  whiteMatterMask[whiteMatterMask == 4 & distance > 10] <- 8

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

  for( j in seq.int( length( segmentations ) ) )
    {
    regionalWmhs <- segmentations[[j]] * antsCopyImageInfo( segmentations[[j]], whiteMatterMask )

    wmhRegionalVolumes <- rep( 0, length( regions ) )
    for( k in seq.int( length( regions ) ) )
      {
      wmhRegionalVolumes[k] <- sum( thresholdImage( regionalWmhs, k, k, 1, 0 ) ) * volumePerVoxel
      }

    singleRow <- c( subjectId, date, seriesId, brainVolume, segmentationNames[j], wmhRegionalVolumes )  

    if( i == 1 && j == 1 )  
      {
      wmhDataFrame <- singleRow
      } else  {
      wmhDataFrame <- rbind( wmhDataFrame, singleRow )  
      }
    }                       
  

  }
cat( "\n" )  

colnames( wmhDataFrame ) <- c( "Subject", "Date", "Series", "BrainVolume", "Algorithm", regions )
rownames( wmhDataFrame ) <- NULL

write.csv( wmhDataFrame, file = "~/Desktop/wmhVolumes.csv", quote = FALSE, row.names = FALSE )

# write.csv( wmhDataFrame, file = paste0( baseDirectory, "/../../Scripts/wmhVolumes.csv" ), quote = FALSE,
#            row.names = FALSE )
