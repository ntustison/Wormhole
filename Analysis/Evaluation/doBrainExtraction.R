library( ANTsRNet )
library( ANTsR ) 

baseDirectory <- "/Users/ntustison/Data/WMH/Nick_compare_ucdWMH/Nifti/"

t1s <- list.files( path = baseDirectory, pattern = "T1.nii.gz", 
  recursive = TRUE, full.names = TRUE )

for( i in seq.int( length( t1s ) ) )  
  {
  cat( "Processing", t1s[i], "\n" )    
  t1 <- antsImageRead( t1s[i] )    
  probMask <- brainExtraction( t1, modality = "t1", verbose = TRUE )
  mask <- thresholdImage( probMask, 0.5, 1, 1, 0 )

  outFile <- sub( "T1", "MASK", t1s[i] )
  cat( "Writing", outFile, "\n" )
  antsImageWrite( mask, outFile )
  }