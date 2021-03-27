library( ANTsR )
library( ANTsRNet )
library( keras )
library( tensorflow )

Sys.setenv( "CUDA_VISIBLE_DEVICES" = "0" )

tf <- tensorflow::tf
keras::backend()$clear_session()
tf$keras$backend$clear_session()
gpus <- tf$config$experimental$list_physical_devices( "GPU" )
tf$config$experimental$set_memory_growth( gpus[[1]], TRUE )
# tf$config$experimental$set_virtual_device_configuration( gpus[[1]],
#   list( tf$config$experimental$VirtualDeviceConfiguration( memory_limit = 1024 ) ) )


baseDirectory <- '/home/ntustison/Data/WMH/'
# baseDirectory <- '/Users/ntustison/Data/WMH/'
scriptsDirectory <- paste0( baseDirectory, 'Scripts/' )
source( paste0( scriptsDirectory, 'batchOctantGenerator.R' ) )

template <- antsImageRead( getANTsXNetData( "croppedMni152" ) )
patchSize <- c( 112L, 112L, 112L )

################################################
#
#  Create the model and load weights
#
################################################

classes <- c( 0:1 )
numberOfClassificationLabels <- length( classes )
imageModalities <- c( "T1", "FLAIR" )
channelSize <- length( imageModalities )

unetModel <- createUnetModel3D( c( patchSize, channelSize ),
   numberOfOutputs = numberOfClassificationLabels, mode = 'classification',
   numberOfLayers = 4, numberOfFiltersAtBaseLayer = 16, dropoutRate = 0.0,
   convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
   weightDecay = 1e-5, nnUnetActivationStyle = FALSE, addAttentionGating = TRUE )

# unetModel <- createResUnetModel3D( c( templateSize, channelSize ),
#   numberOfOutputs = numberOfClassificationLabels, mode = 'classification',
#   numberOfFiltersAtBaseLayer = 8,
#   bottleNeckBlockDepthSchedule = c( 3, 4 ), dropoutRate = 0.0,
#   convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
#   weightDecay = 1e-5 )

brainWeightsFileName <- paste0( scriptsDirectory, "/wmhSegmentationWeights.h5" )
if( file.exists( brainWeightsFileName ) )
  {
  load_model_weights_hdf5( unetModel, filepath = brainWeightsFileName )
  # } else {
  # stop( "Weights file doesn't exist.\n" )
  }


# unet_loss <- weighted_categorical_crossentropy( weights = c( 1, 1000 ) ) # Round1
# unet_loss <- multilabel_dice_coefficient( smoothingFactor = 1.0 ) # Round2
# unet_loss <- multilabel_dice_coefficient( smoothingFactor = 0.1 ) # Round3
unet_loss <- multilabel_surface_loss( dimensionality = 3L )

unetModel %>% compile(
  optimizer = optimizer_adam(),
  loss = unet_loss,
  metrics = c( metric_categorical_crossentropy, 'accuracy' ) )


################################################
#
#  Load the brain data
#
################################################

cat( "Loading brain data.\n" )

t1AdniImages <-
  Sys.glob( paste0( baseDirectory, "ADNI_temp_mni_processed/*/*/*/*/*mnixT1_Preprocessed.nii.gz" ) )
t1Oasis3Images <-
  Sys.glob( paste0( baseDirectory, "MniProcessed/*/*/NIFTI/*mnixT1_Preprocessed.nii.gz" ) )


t1Images <- c( t1AdniImages, t1Oasis3Images )

trainingT1Files <- c()
trainingFlairFiles <- c()
trainingSegFiles <- c()
trainingMaskFiles <- c()

pb <- txtProgressBar( min = 0, max = length( t1Images ), style = 3 )
for( i in seq_len( length( t1Images ) ) )
  {
  setTxtProgressBar( pb, i )

  t1 <- t1Images[i]
  flair <- gsub( "mnixT1_Preprocessed", "mnixT1xFlair", t1 )
  wmh <- gsub( "mnixT1_Preprocessed", "mnixT1_WMH", t1 )
  mask <- gsub( "mnixT1_Preprocessed", "mnixT1_BrainMask", t1 )

  if( ! file.exists( flair ) || ! file.exists( wmh ) || ! file.exists( mask ) )
    {
    next
    # stop( "Mask doesn't exist." )
    }

  # # Do a quick check
#   wmhImage <- antsImageRead( wmh )
#   if( sum( wmhImage ) < 1000 )
#     {
#     cat( "mask: ", wmh, "\n" )
#     next
#     }
#   t1Array <- as.array( antsImageRead( t1 ) )
#   if( any( is.na( t1Array ) ) || ( sum( t1Array ) <= 1000 ) )
#     {
#     cat( "t1: ", t1, "\n" )
#     next
#     }
#   flairArray <- as.array( antsImageRead( flair ) )
#   if( any( is.na( flairArray ) ) || ( sum( flairArray ) <= 1000 ) )
#     {
#     cat( "flair: ", flair, "\n" )
#     next
#     }

  trainingT1Files <- append( trainingT1Files, t1 )
  trainingFlairFiles <- append( trainingFlairFiles, flair )
  trainingSegFiles <- append( trainingSegFiles, wmh )
  trainingMaskFiles <- append( trainingMaskFiles, mask )
  }
cat( "\n" )

cat( "Total training image files: ", length( trainingT1Files ), "\n" )

cat( "\nTraining\n\n" )

###
#
# Set up the training generator
#

batchSize <- 4L

# Split trainingData into "training" and "validation" componets for
# training the model.

numberOfData <- length( trainingT1Files )
sampleIndices <- sample( numberOfData )

validationSplit <- floor( 0.8 * numberOfData )
trainingIndices <- sampleIndices[1:validationSplit]
numberOfTrainingData <- length( trainingIndices )
validationIndices <- sampleIndices[( validationSplit + 1 ):numberOfData]
numberOfValidationData <- length( validationIndices )

###
#
# Run training
#

track <- unetModel %>% fit_generator(
  generator = batchGenerator( batchSize = batchSize,
                              patchSize = patchSize,
                              template = template,
                              t1s = trainingT1Files[trainingIndices],
                              flairs = trainingFlairFiles[trainingIndices],
                              brainMasks = trainingMaskFiles[trainingIndices],
                              segmentationImages = trainingSegFiles[trainingIndices],
                              segmentationLabels = classes,
                              doRandomContralateralFlips = FALSE,
                              doDataAugmentation = FALSE
                            ),
  steps_per_epoch = 48L,
  epochs = 200L,
  validation_data = batchGenerator( batchSize = batchSize,
                                    patchSize = patchSize,
                                    template = template,
                                    t1s = trainingT1Files[validationIndices],
                        				    flairs = trainingFlairFiles[validationIndices],
				                            brainMasks = trainingMaskFiles[validationIndices],
                                    segmentationImages = trainingSegFiles[validationIndices],
				                            segmentationLabels = classes,
                                    doRandomContralateralFlips = FALSE,
                                    doDataAugmentation = FALSE
                                  ),
  validation_steps = 12L,
  callbacks = list(
    callback_model_checkpoint( brainWeightsFileName,
      monitor = 'val_loss', save_best_only = TRUE, save_weights_only = TRUE,
      verbose = 1, mode = 'auto' ),
     callback_reduce_lr_on_plateau( monitor = 'val_loss', factor = 0.5,
       verbose = 1, patience = 10, mode = 'auto' ),
    callback_early_stopping( monitor = 'val_loss', min_delta = 0.001,
      patience = 20 )
  )
)

save_model_weights_hdf5( unetModel, filepath = brainWeightsFileName )
