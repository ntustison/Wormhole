#/usr/bin/perl -w

use strict;

use Cwd 'realpath';
use File::Find;
use File::Basename;
use File::Path;
use File::Spec;

my $baseDirectory = "/Users/ntustison/Data/WMH/";

my @suffixList = ( ".nii.gz" );
my $numberOfThreads = 4;


my @subjectDirs = <${baseDirectory}/ADNI/ADNI_temp/???_S_????/>;


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

  my $dkt = "${outputPrefix}_ants_dkt.nii.gz";
  my $dktLobes = "${outputPrefix}_ants_dktLobes.nii.gz";

  my $doCluster = 0;

  my $commandFile = "${path}/dktCommand.sh";
  open( FILE, ">${commandFile}" );
  print FILE "#!/bin/sh\n";
  print FILE "export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${numberOfThreads}\n\n";

  print FILE "\n\n";
  # print FILE "rm -f $mask $segmentation $wmh\n";

  if( ! -e $dkt || ! -e $dktLobes )
    {
    print FILE "python3 /Users/ntustison/Data/Papers/Wormhole/Analysis/Pipeline/do_dkt_processing.py $t1 $outputPrefix $numberOfThreads\n";
    $doCluster = 1;
    }

  close( FILE );

  print "$commandFile \n";

  if( $doCluster == 1 )
    {
    # my @qargs = ( 'sbatch',
    #               "--job-name=dkt_${count}",
    #               '--partition=free',
    #               '--nodes=1',
    #               '--ntasks=1',
    #               "--cpus-per-task=${numberOfThreads}",
    #               '--account=ntustiso',
    #               # '--mem=12gb`',
    #               # '--time=8:00:00',
    #               $commandFile ); 
    # system( @qargs ) == 0 || die "sbatch\n";
 
    system( "sh $commandFile" );
    $count++;
    }
#  if( $count > 1 )
#    {
#    die "Count\n";
#    }
  }

print "Total: $count\n";
