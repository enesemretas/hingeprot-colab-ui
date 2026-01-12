#!/usr/bin/perl -w

use strict;
use File::Copy;

my $home="/home/appserv/web/prc/hingeprot/bin";

if ($#ARGV != 0) {
  print "runNMA.pl <PDB_file>\n";
  exit;
}

my $pdb = $ARGV[0];
my $pdbCode = $ARGV[0];

#trim ".pdb" extension if exists
my @tmp=split('\.',$pdbCode);
if($tmp[$#tmp] =~ /^pdb$/i && $#tmp==1) {
  $pdbCode = $tmp[0];
}

if(! -e $pdb) {
  print "Can't find file $pdb\n";
  exit;
}
my $filesize = -s "$pdb";
if($filesize == 0) {
  print "Empty file $pdb\n";
  exit;
}

#print "$pdbCode\n";
copy("$home/rescale","rescale");
copy("$home/gnmcutoff","gnmcutoff");
copy("$home/anmcutoff","anmcutoff");
copy("$home/rescale","rescale");
copy($pdb, "pdb");
`$home/read; $home/gnmc; $home/anm2; $home/useblz; $home/anm3; $home/extract; $home/coor2pdb;` ;

rename("coor1.mds12", "$pdbCode.cor1A");
rename("coor2.mds12", "$pdbCode.cor1B");
rename("coor3.mds12", "$pdbCode.cor2A");
rename("coor4.mds12", "$pdbCode.cor2B");
rename("slowmodes", "$pdbCode.slowmodes");
rename("slow12avg", "$pdbCode.slow12avg");
rename("crosscorr", "$pdbCode.crosscorr");
rename("alpha.cor", "$pdbCode.CA");
rename("1coor", "$pdbCode.anm1vector");
rename("2coor", "$pdbCode.anm2vector");
rename("3coor", "$pdbCode.anm3vector");
rename("4coor", "$pdbCode.anm4vector");
rename("5coor", "$pdbCode.anm5vector");
rename("6coor", "$pdbCode.anm6vector");
rename("7coor", "$pdbCode.anm7vector");
rename("8coor", "$pdbCode.anm8vector");
rename("9coor", "$pdbCode.anm9vector");
rename("10coor", "$pdbCode.anm10vector");
rename("11coor", "$pdbCode.anm11vector");
rename("12coor", "$pdbCode.anm12vector");
rename("13coor", "$pdbCode.anm13vector");
rename("14coor", "$pdbCode.anm14vector");
rename("15coor", "$pdbCode.anm15vector");
rename("16coor", "$pdbCode.anm16vector");
rename("17coor", "$pdbCode.anm17vector");
rename("18coor", "$pdbCode.anm18vector");
rename("19coor", "$pdbCode.anm19vector");
rename("20coor", "$pdbCode.anm20vector");
rename("21coor", "$pdbCode.anm21vector");
rename("22coor", "$pdbCode.anm22vector");
rename("23coor", "$pdbCode.anm23vector");
rename("24coor", "$pdbCode.anm24vector");
rename("25coor", "$pdbCode.anm25vector");
rename("26coor", "$pdbCode.anm26vector");
rename("27coor", "$pdbCode.anm27vector");
rename("28coor", "$pdbCode.anm28vector");
rename("29coor", "$pdbCode.anm29vector");
rename("30coor", "$pdbCode.anm30vector");
rename("31coor", "$pdbCode.anm31vector");
rename("32coor", "$pdbCode.anm32vector");
rename("33coor", "$pdbCode.anm33vector");
rename("34coor", "$pdbCode.anm34vector");
rename("35coor", "$pdbCode.anm35vector");
rename("36coor", "$pdbCode.anm36vector");

rename("gnm1anmvector", "$pdbCode.1vector");
rename("gnm2anmvector", "$pdbCode.2vector");
rename("hinges", "$pdbCode.hinge");
rename("crosscorrslow1", "$pdbCode.crossslow1");
rename("crosscorrslow2", "$pdbCode.crossslow2");
rename("newcoordinat.mds", "$pdbCode.new");
rename("anm_length", "$pdbCode.anm1D");
rename("mod1","$pdbCode.modeanimation1");
rename("mod2","$pdbCode.modeanimation2");

`zip ANM-MSF.zip $pdbCode.anm?vector $pdbCode.anm10vector`;
`zip ANM-MSFa.zip $pdbCode.anm*vector`;

`$home/processHinges $pdbCode.new $pdbCode.hinge $pdbCode.anm1vector $pdbCode.anm2vector 15 14.0`;
rename("$pdbCode.new.moved1.pdb", "anment1.pdb");
rename("$pdbCode.new.moved2.pdb", "anment2.pdb");
`$home/processHinges $pdbCode.new $pdbCode.hinge $pdbCode.anm3vector $pdbCode.anm4vector 15 14.0`;
rename("$pdbCode.new.moved1.pdb", "anment3.pdb");
rename("$pdbCode.new.moved2.pdb", "anment4.pdb");
`$home/processHinges $pdbCode.new $pdbCode.hinge $pdbCode.anm5vector $pdbCode.anm6vector 15 14.0`;
rename("$pdbCode.new.moved1.pdb", "anment5.pdb");
rename("$pdbCode.new.moved2.pdb", "anment6.pdb");
`$home/processHinges $pdbCode.new $pdbCode.hinge $pdbCode.anm7vector $pdbCode.anm8vector 15 14.0`;
rename("$pdbCode.new.moved1.pdb", "anment7.pdb");
rename("$pdbCode.new.moved2.pdb", "anment8.pdb");
`$home/processHinges $pdbCode.new $pdbCode.hinge $pdbCode.anm9vector $pdbCode.anm10vector 15 14.0`;
rename("$pdbCode.new.moved1.pdb", "anment9.pdb");
rename("$pdbCode.new.moved2.pdb", "anment10.pdb");
`$home/processHinges $pdbCode.new $pdbCode.hinge $pdbCode.anm11vector $pdbCode.anm12vector 15 14.0`;
rename("$pdbCode.new.moved1.pdb", "anment11.pdb");
rename("$pdbCode.new.moved2.pdb", "anment12.pdb");
`$home/processHinges $pdbCode.new $pdbCode.hinge $pdbCode.anm13vector $pdbCode.anm14vector 15 14.0`;
rename("$pdbCode.new.moved1.pdb", "anment13.pdb");
rename("$pdbCode.new.moved2.pdb", "anment14.pdb");
`$home/processHinges $pdbCode.new $pdbCode.hinge $pdbCode.anm15vector $pdbCode.anm16vector 15 14.0`;
rename("$pdbCode.new.moved1.pdb", "anment15.pdb");
rename("$pdbCode.new.moved2.pdb", "anment16.pdb");
`$home/processHinges $pdbCode.new $pdbCode.hinge $pdbCode.anm17vector $pdbCode.anm18vector 15 14.0`;
rename("$pdbCode.new.moved1.pdb", "anment17.pdb");
rename("$pdbCode.new.moved2.pdb", "anment18.pdb");
`$home/processHinges $pdbCode.new $pdbCode.hinge $pdbCode.anm19vector $pdbCode.anm20vector 15 14.0`;
rename("$pdbCode.new.moved1.pdb", "anment19.pdb");
rename("$pdbCode.new.moved2.pdb", "anment20.pdb");
`$home/processHinges $pdbCode.new $pdbCode.hinge $pdbCode.anm21vector $pdbCode.anm22vector 15 14.0`;
rename("$pdbCode.new.moved1.pdb", "anment21.pdb");
rename("$pdbCode.new.moved2.pdb", "anment22.pdb");
`$home/processHinges $pdbCode.new $pdbCode.hinge $pdbCode.anm23vector $pdbCode.anm24vector 15 14.0`;
rename("$pdbCode.new.moved1.pdb", "anment23.pdb");
rename("$pdbCode.new.moved2.pdb", "anment24.pdb");
`$home/processHinges $pdbCode.new $pdbCode.hinge $pdbCode.anm25vector $pdbCode.anm26vector 15 14.0`;
rename("$pdbCode.new.moved1.pdb", "anment25.pdb");
rename("$pdbCode.new.moved2.pdb", "anment26.pdb");
`$home/processHinges $pdbCode.new $pdbCode.hinge $pdbCode.anm27vector $pdbCode.anm28vector 15 14.0`;
rename("$pdbCode.new.moved1.pdb", "anment27.pdb");
rename("$pdbCode.new.moved2.pdb", "anment28.pdb");
`$home/processHinges $pdbCode.new $pdbCode.hinge $pdbCode.anm29vector $pdbCode.anm30vector 15 14.0`;
rename("$pdbCode.new.moved1.pdb", "anment29.pdb");
rename("$pdbCode.new.moved2.pdb", "anment30.pdb");
`$home/processHinges $pdbCode.new $pdbCode.hinge $pdbCode.anm31vector $pdbCode.anm32vector 15 14.0`;
rename("$pdbCode.new.moved1.pdb", "anment31.pdb");
rename("$pdbCode.new.moved2.pdb", "anment32.pdb");
`$home/processHinges $pdbCode.new $pdbCode.hinge $pdbCode.anm33vector $pdbCode.anm34vector 15 14.0`;
rename("$pdbCode.new.moved1.pdb", "anment33.pdb");
rename("$pdbCode.new.moved2.pdb", "anment34.pdb");
`$home/processHinges $pdbCode.new $pdbCode.hinge $pdbCode.anm35vector $pdbCode.anm36vector 15 14.0`;
rename("$pdbCode.new.moved1.pdb", "anment35.pdb");
rename("$pdbCode.new.moved2.pdb", "anment36.pdb");

`$home/processHinges $pdbCode.new $pdbCode.hinge $pdbCode.1vector $pdbCode.2vector 15 14.0`;
#rename("$pdbCode.new.hinges", "hingeain");
#rename("$pdbCode.CA", "CA");
#`$home/hingeaa`;
#rename("CA", "$pdbCode.CA");
#rename("hingeout", "$pdb.hinges");
rename("$pdbCode.new.loops", "$pdb.loops");
copy("$pdbCode.new.moved1.pdb", "modeent1");
copy("$pdbCode.new.moved2.pdb", "modeent2");
`$home/splitter`;
unlink <modeent*>;
rename("$pdbCode.new.moved1.pdb", "$pdbCode.mode1.ent");
rename("$pdbCode.new.moved2.pdb", "$pdbCode.mode2.ent");
rename("sortedeigen","$pdbCode.eigengnm");
rename("eigenanm","$pdbCode.eigenanm");

#my @filelist = ("coordinates","upperhessian", "sloweigenvectors");

#unlink @filelist;
my @paramlist =("gnmcutoff","anmcutoff","rescale");
#unlink <.*>; #hidden files
#unlink <fort*>;
