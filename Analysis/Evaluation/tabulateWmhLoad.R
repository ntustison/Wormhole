library( ANTsRNet )
library( ANTsR ) 
library( GGally )


baseDirectory <- "/Users/ntustison/Data/WMH/ADNI/ADNI_temp"

octants <- list.files( path = baseDirectory, pattern = "EwDavidOctantWmhSegmentation.nii.gz", 
  recursive = TRUE, full.names = TRUE )

wmhDataFrame <- data.frame()

pb <- txtProgressBar( min = 0, max = length( octants ), style = 3 )
# for( i in seq.int( 10 ) )  
for( i in seq.int( length( octants ) ) )  
  {
  tokens <- strsplit( octants[i], "/" )   
  subjectId <- tokens[[1]][8]
  date <- tokens[[1]][10]
  seriesId <- tokens[[1]][11]

  setTxtProgressBar( pb, i )  
  brainMaskT1 <- sub( "FLAIR_EwDavidOctantWmhSegmentation", "BrainMask", octants[i] )  
  t1xflairXfrm <- sub( "FLAIR_EwDavidOctantWmhSegmentation.nii.gz", "T1xFLAIR0GenericAffine.mat", octants[i] )  

  octant <- antsImageRead( octants[i] )
  sliceWise7mb <- antsImageRead( sub( "EwDavidOctantWmhSegmentation", "EwDavidSlicewiseWmhSegmentation_7mb", octants[i] ) )
  sliceWise300mb <- antsImageRead( sub( "EwDavidOctantWmhSegmentation", "EwDavidSlicewiseWmhSegmentation_300mb", octants[i] ) )
  sliceWiseT1Only <- antsImageRead( sub( "EwDavidOctantWmhSegmentation", "EwDavidSlicewiseWmhSegmentation_T1Only", octants[i] ) )
  combined <- thresholdImage( antsCopyImageInfo( sliceWise300mb, octant ) + sliceWise300mb, 0, 0, 0, 1 )

  sysu <- antsImageRead( sub( "EwDavidOctantWmhSegmentation", "SysuWmhSegmentation", octants[i] ) )
  sysuAxial <- antsImageRead( sub( "EwDavidOctantWmhSegmentation", "SysuWmhAxialSegmentation", octants[i] ) )
  
  ucd <- thresholdImage( antsImageRead( sub( "ants_FLAIR_EwDavidOctantWmhSegmentation", "itk_FLAIR_WMH_Native", octants[i] ) ), 0, 0, 0, 1 )


  # t1Combined <- antsImageRead( sub( "FLAIR_EwDavidOctantWmhSegmentation", "T1_combinedWMH", octants[i] ) )
  # t1Combined <- antsApplyTransforms( octant, t1Combined, 
  #   transformlist = c( t1xflairXfrm ), whichtoinvert = c( TRUE ),
  #   interpolator = "nearestNeighbor" )
  
  brainMask <- antsApplyTransforms( octant, brainMaskT1, 
    transformlist = c( t1xflairXfrm ), whichtoinvert = c( TRUE ),
    interpolator = "nearestNeighbor" )

  volumePerVoxel <- prod( antsGetSpacing( brainMask ) )
  brainVolume <- sum( brainMask ) * volumePerVoxel

  segmentations <- list( octant, 
                         sliceWise7mb,
                         sliceWise300mb,
                         sliceWiseT1Only,
                         combined,
                         sysu, 
                         sysuAxial,
                         ucd )

  wmhVolumes <- rep( 0, length( segmentations ) )
  for( j in seq.int( length( segmentations ) ) )
    {
    brainMaskTmp <- antsCopyImageInfo( segmentations[[j]], brainMask )
    wmh <- segmentations[[j]] * brainMaskTmp
    wmhVolumes[j] <- sum( wmh ) * volumePerVoxel
    }                       
  singleRow <- c( subjectId, date, seriesId, brainVolume, wmhVolumes )

  if( i == 1 )  
    {
    wmhDataFrame <- singleRow
    } else  {
    wmhDataFrame <- rbind( wmhDataFrame, singleRow )  
    }
  }
cat( "\n" )  

colnames( wmhDataFrame ) <- c( "Subject", "Date", "Series", "BrainVolume", "Octant", "Slicewise7mb", "Slicewise300mb", "SlicewiseT1Only", "Combined", "Sysu", "SysuAxial", "UCD" )
rownames( wmhDataFrame ) <- NULL

write.csv( wmhDataFrame, file = paste0( baseDirectory, "/../../Scripts/wmhVolumes.csv" ), quote = FALSE,
           row.names = FALSE )
