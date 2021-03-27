batchGenerator <- function( batchSize = 32L,
                            patchSize = c( 64, 64, 64 ),
                            template = NULL,
                            t1s = NULL,
                            flairs = NULL,
                            brainMasks = NULL,
                            segmentationImages = NULL,
			                      segmentationLabels = NULL,
                            doRandomContralateralFlips = TRUE,
                            doDataAugmentation = TRUE
 )
{                                     

  if( is.null( template ) )
    {
    stop( "No reference template specified." )
    }
  if( is.null( t1s ) || is.null( flairs ) )
    {
    stop( "Input images must be specified." )
    }
  if( is.null( segmentationImages ) )
    {
    stop( "Input masks must be specified." )
    }
  if( is.null( segmentationLabels ) )
    {
    stop( "segmentationLabels must be specified." )
    }

  strideLength <- dim( template ) - patchSize

  currentPassCount <- 0L

  function()
    {
    # Shuffle the data after each complete pass

    if( ( currentPassCount + batchSize ) >= length( t1s ) )
      {
      # shuffle the source data
      sampleIndices <- sample( length( t1s ) )
      t1s <- t1s[sampleIndices]
      flairs <- flairs[sampleIndices]
      brainMasks <- brainMasks[sampleIndices]
      segmentationImages <- segmentationImages[sampleIndices]

      currentPassCount <- 0L
      }

    batchIndices <- currentPassCount + 1L:batchSize

    batchT1s <- t1s[batchIndices]
    batchFlairs <- flairs[batchIndices]
    batchBrainMasks <- brainMasks[batchIndices]
    batchWmhs <- segmentationImages[batchIndices]

    X <- array( data = 0, dim = c( batchSize, patchSize, 1 ) )    
    Y <- array( data = 0L, dim = c( batchSize, patchSize ) )

    currentPassCount <<- currentPassCount + batchSize

    # pb <- txtProgressBar( min = 0, max = batchSize, style = 3 )
    
    count <- 1
    while( count <= batchSize )
      {
      i <- sample.int( batchSize, 1 ) 	      
  
      # setTxtProgressBar( pb, i )

      t1 <- NULL 
      flair <- NULL
      wmh <- NULL

      mask <- antsImageRead( batchBrainMasks[i] )
      batchT1 <- antsImageRead( batchT1s[i] ) * mask
      batchFlair <- antsImageRead( batchFlairs[i] ) * mask
      batchWmh <- antsImageRead( batchWmhs[i] )

      # antsImageWrite( batchT1, "t1.nii.gz" );
      # antsImageWrite( batchFlair, "flair.nii.gz" );
      # antsImageWrite( batchWmh, "wmh.nii.gz" );
      # stop( "HERE" )

      if( doRandomContralateralFlips && sample( c( TRUE, FALSE ) ) )
        {
        t1A <- as.array( batchT1 )
	t1 <- as.antsImage( t1A[dim( t1A )[1]:1,,], reference = batchT1 )
        flairA <- as.array( batchFlair )
        flair <- as.antsImage( flairA[dim( flairA )[1]:1,,], reference = batchFlair )
        wmhA <- as.array( batchWmh )
        wmh <- as.antsImage( wmhA[dim( wmhA )[1]:1,,], reference = batchWmh )
        } else {
        t1 <- batchT1  
	      flair <- batchFlair
        wmh <- batchWmh
        }
      
      warpedT1 <- t1
      warpedFlair <- flair
      warpedWmh <- wmh
      
      wmhPatches <- NULL
      t1Patches <- NULL
      flairPatches <- NULL

      if( doDataAugmentation == TRUE )
        {
        dataAugmentation <- 
          randomlyTransformImageData( template, 
          list( list( warpedT1, warpedFlair ) ),
          list( warpedWmh ),
          numberOfSimulations = 1, 
          transformType = 'affineAndDeformation', 
          sdAffine = 0.01,
          deformationTransformType = "bspline",
          numberOfRandomPoints = 1000,
          sdNoise = 2.0,
          numberOfFittingLevels = 4,
          meshSize = 1,
          sdSmoothing = 4.0,
          inputImageInterpolator = 'linear',
          segmentationImageInterpolator = 'nearestNeighbor' )

        simulatedT1 <- dataAugmentation$simulatedImages[[1]][[1]]
        simulatedT1 <- ( simulatedT1 - mean( simulatedT1 ) ) / sd( simulatedT1 )
        simulatedFlair <- dataAugmentation$simulatedImages[[1]][[2]]
        simulatedFlair <- ( simulatedFlair - mean( simulatedFlair ) ) / sd( simulatedFlair )

        simulatedWmh <- dataAugmentation$simulatedSegmentationImages[[1]]

        # antsImageWrite( simulatedImage, paste0( "./TempData/simBrainImage", i, ".nii.gz" ) )
        # antsImageWrite( simulatedSegmentationImage, paste0( "./TempData/simSegmentationImage", i, ".nii.gz" ) )

        t1Patches <- extractImagePatches( simulatedT1, patchSize, maxNumberOfPatches = "all",
					  strideLength = strideLength, randomSeed = NULL,
					  returnAsArray = TRUE )
        flairPatches <- extractImagePatches( simulatedFlair, patchSize, maxNumberOfPatches = "all",
                                             strideLength = strideLength, randomSeed = NULL,
                                             returnAsArray = TRUE )
        wmhPatches <- extractImagePatches( simulatedWmh, patchSize, maxNumberOfPatches = "all",
                                           strideLength = strideLength, randomSeed = NULL,
                                           returnAsArray = TRUE )	

        } else {
        warpedT1 <- ( warpedT1 - mean( warpedT1 ) ) / sd( warpedT1 )
        warpedFlair <- ( warpedFlair - mean( warpedFlair ) ) / sd( warpedFlair )

        # antsImageWrite( warpedImage, paste0( "./TempData/wBrainImage", i, ".nii.gz" ) )
        # antsImageWrite( warpedSegmentationImage, paste0( "./TempData/wSegmentationImage", i, ".nii.gz" ) )

        t1Patches <- extractImagePatches( warpedT1, patchSize, maxNumberOfPatches = "all",
                                          strideLength = strideLength, randomSeed = NULL,
                                          returnAsArray = TRUE )
        flairPatches <- extractImagePatches( warpedFlair, patchSize, maxNumberOfPatches = "all",
                                             strideLength = strideLength, randomSeed = NULL,
                                             returnAsArray = TRUE )
        wmhPatches <- extractImagePatches( warpedWmh, patchSize, maxNumberOfPatches = "all",
                                           strideLength = strideLength, randomSeed = NULL,
                                           returnAsArray = TRUE )
      	}

      whichOctant <- sample.int( 8, 1 )
      # cat( i, ", ", whichOctant, ":")  
      if( sum( wmhPatches[whichOctant,,,] ) >= 1000 )
        {
        # cat( "  Yes!! \n")  
        X[count,,,,1] <- t1Patches[whichOctant,,,]
        # X[count,,,,2] <- flairPatches[whichOctant,,,]
        Y[count,,,] <- wmhPatches[whichOctant,,,]

        # antsImageWrite( as.antsImage( drop( t1Patches[whichOctant,,,] ) ), "t1.nii.gz" );
        # antsImageWrite( as.antsImage( drop( flairPatches[whichOctant,,,] ) ), "flair.nii.gz" );
        # antsImageWrite( as.antsImage( drop( wmhPatches[whichOctant,,,] ) ), "wmh.nii.gz" );
        # stop( "HERE" )

        count <- count + 1  
        } # else { 
        # cat( "  No!! \n")  
        # }
      }
    # stop( "Done testing.")  
    # cat( "\n" )

    encodedY <- encodeUnet( Y, segmentationLabels )

    return( list( X, encodedY ) )
    }
}
