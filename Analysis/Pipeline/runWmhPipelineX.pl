#/usr/bin/perl -w

use strict;

use Cwd 'realpath';
use File::Find;
use File::Basename;
use File::Path;
use File::Spec;

my $baseDirectory = "/pub/ntustiso/Data/WMH/";

my @suffixList = ( ".nii.gz" );
my $numberOfThreads = 1;

my $ANTSPATH = "/data/homezvol1/ntustiso/Pkg/ANTs/bin/ANTS-build/Examples/";

my @subjectDirs = <${baseDirectory}/ADNI_temp/???_S_????/>;


my $count = 0;

for( my $i = 0; $i < @subjectDirs; $i++ )
  {

  my @tmp = ( <$subjectDirs[$i]/*FLAIR/*/*/*_I??????.nii.gz>, <$subjectDirs[$i]/*FLAIR/*/*/*_I???????.nii.gz> );
  if( @tmp == 0 )
    {
    print "FLAIR:  $subjectDirs[$i]\n";
    next;
    }
  my $flair = $tmp[0];

  my @tmp2 = ( <$subjectDirs[$i]/*MPRAGE/*/*/*_I??????.nii.gz>, <$subjectDirs[$i]/*MPRAGE/*/*/*_I???????.nii.gz> );
  if( @tmp2 == 0 )
    {
    @tmp2 = ( <$subjectDirs[$i]/*SPGR/*/*/*_I??????.nii.gz>, <$subjectDirs[$i]/*SPGR/*/*/*_I???????.nii.gz> );
    }
  if( @tmp2 == 0 )
    {
    print "MPRAGE: $subjectDirs[$i]\n";
    next;
    }
  my $t1 = $tmp2[0];

  my ( $filename, $path, $suffix ) = fileparse( $t1, @suffixList );

  my $outputPrefix = "${path}/${filename}";

  my $t1xflair = "${outputPrefix}_ants_T1xFLAIR.nii.gz";
  my $t1xflairXfrm = "${outputPrefix}_ants_T1xFLAIR0GenericAffine.mat";
  my $mask = "${outputPrefix}_ants_BrainMask.nii.gz";
  my $segmentation = "${outputPrefix}_ants_BrainSegmentation.nii.gz";
  my $wmh = "${outputPrefix}_ants_SysuWmhSegmentation.nii.gz";

  my $commandFile = "${path}/wmhCommand.sh";
  open( FILE, ">${commandFile}" );
  print FILE "#!/bin/sh\n";
  print FILE "export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${numberOfThreads}\n\n";

  print FILE "\n\n";
  # print FILE "rm -f $mask $segmentation $wmh\n";
  if( ! -e $t1xflair || ! -e $segmentation || ! -e $wmh )
    {
    print FILE "python3 ${baseDirectory}/Scripts/do_ants_processing.py $t1 $flair $outputPrefix $numberOfThreads\n";
    }
  if( ! -e $mask )
    {
    print FILE "${ANTSPATH}/ThresholdImage 3 $segmentation $mask 0 0 0 1\n";
    }

  my $ucd = "${outputPrefix}_itk_T1_Stripped_corrSmTP_segmentedWMT-Z4T.nii.gz";
  if( ! -e $ucd )
    {
    # Need to do the following because the ADNI niftis are seemingly
    # incompatible with the UCD algorithm

    my $itkt1 = "${outputPrefix}_itk_T1.nii.gz";
    my $itkflair = "${outputPrefix}_itk_FLAIR.nii.gz";
    if( ! -e $itkt1 )
      {
      print FILE "${ANTSPATH}/ImageMath 3 $itkt1 m $t1 1\n";
      }
    if( ! -e $itkflair )
      {
      print FILE "${ANTSPATH}/ImageMath 3 $itkflair m $flair 1\n";
      }
    print FILE "python3 /data/homezvol1/ntustiso/Pkg/UCDWMHSegmentation-1.3/ucd_wmh_segmentation/ucd_wmh_segmentation.py $itkt1 $mask $itkflair --delete-temporary\n";
    print FILE "rm -f $itkflair $itkt1\n";
    }
  close( FILE );

  print "$commandFile \n";

  if( ! -e $ucd || ! -e $mask || ! -e $segmentation || ! -e $wmh )
    {
    my @qargs = ( 'sbatch',
                  "--job-name=wmh_${count}",
                  '--partition=free',
                  '--nodes=1',
                  '--ntasks=1',
                  "--cpus-per-task=${numberOfThreads}",
                  '--account=ntustiso',
                  # '--mem=12gb`',
                  # '--time=8:00:00',
                  $commandFile ); 
    system( @qargs ) == 0 || die "sbatch\n";
    $count++;
    }
  # if( $count > 1 )
  #  {
  #  die "Count\n";
  #  }
  }

print "Total: $count\n";
