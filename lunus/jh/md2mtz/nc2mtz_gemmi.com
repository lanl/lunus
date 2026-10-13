#! /bin/tcsh -f
#
# convert nc file to an mtz file
#
#
set smallSG = ""
set reso = 1
set rate = 2
set B = 10
set super_mult = ( 1 1 1 )
set ncfile = ""
set topfile = xtal.prmtop
set paddedparm = padded.parm7
set orignames = ""
set Bfac_file = ""
set auto_Bfac_file = rmsd2B.pdb

set outtraj = trajectory
set outmap = avg.map
set outfile = avg.mtz

set Bscale = 1
set Boffset = 0
set maxB = 100
set minB = 2

set wrap = 0
set keeptraj = 0
set domaps = 1
set addmaps = 1
set domtzs = 0
set addmtzs = 0

set debug = 0


set tempfile = /dev/shm/${USER}/temp_$$_traj/

set pdir = `dirname $0`

set CPUs = `grep proc /proc/cpuinfo | wc -l | awk '{print int($1/4)}'`
if( "$CPUs" == "" ) set CPUs = 1

set thishost = `hostname -s`

set startepoch = `msdate.com | awk '{print $7}'`

# read command line
foreach Arg ( $* )
    set arg = `echo $Arg | awk '{print tolower($0)}'`
    set assign = `echo $arg | awk '{print ( /=/ )}'`
    set Key = `echo $Arg | awk -F "=" '{print $1}'`
    set Val = `echo $Arg | awk '{print substr($0,index($0,"=")+1)}'`
    set Csv = `echo $Val | awk 'BEGIN{RS=","} {print}'`
    set key = `echo $Key | awk '{print tolower($1)}'`
    set num = `echo $Val | awk '{print $1+0}'`
    set int = `echo $Val | awk '{print int($1+0)}'`

    if( $assign ) then
      # re-set any existing variables
      set test = `set | awk -F "\t" '{print $1}' | egrep "^${Key}"'$' | wc -l`
      if ( $test ) then
          set $Key = $Val
          echo "$Key = $Val"
          continue
      endif
      # synonyms
      if( $arg == md_mult ) set super_mult = ( $Val )
    else
      # no equal sign
      if("$arg" =~ [pcifrh][1-6]*) set smallSG = `echo $arg | awk '{print toupper($0)}'`
      if("$key" =~ *.nc ) set ncfile = "$Arg"
    endif
    if("$key" == "debug") set debug = "$Val"
end

if( $domtzs ) set domaps = 1
if( $addmaps ) set domaps = 1
if( $addmtzs ) set domtzs = 1

mkdir -p ${tempfile}
if($status) then
  echo "WARNING: reverting to local tempfile"
  set tempfile = /dev/shm/${USER}/nc2mtz_$$_
  mkdir -p ${tempfile}
  if( $status ) then
    set tempfile = ./nc2mtz_$$_
    mkdir -p ${tempfile}
  endif
endif

# use cluster or not?
# cannot migrate hosts because of temp files
set test = `sinfo -h -n $thishost |& egrep -v "drain|n/a" | awk '$2=="up"' | wc -l`
if ( $test ) then
  echo "using slurm"
  set srun = "srun -w $thishost"
  set test = `echo $tempfile | awk '{print ( ! /\/dev\/shm/ )}'`
  if( $test ) then
    echo "full cluster"
    set srun = "srun"
  endif
else
  set srun = ""
endif

set t = $tempfile

if( "$smallSG" == "" ) set smallSG = P1
set smallSGnum = `awk -v SG=$smallSG '$4 == SG && $1 < 500 {print $1}' $CLIBD/symop.lib | head -1`
set smallSG = `awk -v num=$smallSGnum '$1==num && NF>5{print $4}' ${CLIBD}/symop.lib`
if("$smallSG" == "") then
    set BAD = "bad space group."
    goto exit
endif

set super_mult = `echo $super_mult | awk '{gsub("[,x]"," ");print}'`
if( "$super_mult" == "" ) set super_mult = ( 1 1 1 )
while ( $#super_mult != 3 )
  set super_mult = ( $super_mult $super_mult[$#super_mult] )
end
set nsymops = `awk -v SG=$smallSG '$4 == SG && $1 < 500 {print $2}' $CLIBD/symop.lib | head -1`

cat << EOF
ncfile = $ncfile
smallSG = $smallSG
super_mult = $super_mult
tempfile = $t
EOF


mkdir -p ${t}
rm -f $outtraj > /dev/null
ln -sf ${t} $outtraj


dowrap:
set image = ""
if( $wrap ) set image = "image byatom"
cat << EOF >! ${t}cpptraj.in
$image
strip :WAT,HOH@Y1,EPW
outtraj ${outtraj}/md.pdb pdb multi pdbv3 keepext sg "P 1"
go
EOF
# outtraj trajectory/md.pdb pdb multi pdbv3 keepext sg "P 1" onlyframes ${s}-${f}

again:
echo "cpptraj..."
echo list |\
cpptraj -y $ncfile -p $topfile >&! ${t}cpptraj.log
if($status && -e "$paddedparm" && ! $?RESIZE) then
  set RESIZE
  echo "making new parm file from $paddedparm"
  mv ${t}cpptraj.log ${t}cpptraj_error1.log
  set rstatoms = `awk '/Error: Number of atoms in /{gsub("[)(]","");for(i=NF;i>3;--i)if($i+0>0)print $i;exit}' ${t}cpptraj_error1.log | head -n 1`
  if( "$rstatoms" == "" ) then
   set rstatoms = `awk '/Error: Number of atoms in /{gsub("[)(]","");print $(NF-2);exit}' ${t}cpptraj_error1.log`
  endif
  set maxatoms = `echo list | cpptraj -p $paddedparm | awk '$4=="atoms,"{print $3}' | head -n 1`
  set stripmask = `echo $rstatoms $maxatoms | awk '{print $1+1"-"$2}'`
  set topfile = ${t}resized.parm7
  echo "new parmfile: $topfile"
  cpptraj -p $paddedparm << EOF >&! ${t}strip1.log
  parmstrip @$stripmask
  parmwrite out $topfile
EOF
  goto again
endif

set nframes = `awk '/Coordinate processing will occur on/{print $6}' ${t}cpptraj.log`
echo "$nframes frames in $ncfile"
if( $nframes < 10 ) then
  set chunks = 1
else
  set chunks = $CPUs
endif
if( $chunks > ( $nframes / 2 ) ) @ chunks = ( $nframes / 2 )
if( $chunks < 1 ) set chunks = 1

#set t0 = `msdate.com | awk '{print $NF}'`
@ chunksize = ( $nframes / $chunks )
foreach chunk ( `seq 1 $chunks` )
  set s = `echo $chunk $chunksize | awk '{print ($1-1)*$2+1}'`
  set f = `echo $s $chunksize | awk '{print $1+$2-1}'`
  if( $chunk == $chunks ) set f = $nframes
  echo "chunk $chunk is $s - $f"
  cat << EOF >! ${t}cpptraj_${chunk}.in
$image
strip :WAT,HOH@Y1,EPW
outtraj ${outtraj}/md.pdb pdb multi pdbv3 keepext sg "P 1" onlyframes ${s}-${f}
go
EOF
  cat ${t}cpptraj_${chunk}.in | $srun cpptraj -y $ncfile -p $topfile >&! ${t}cpptraj_${chunk}.log &
end
wait
#set dt = `msdate.com $t0 | awk '{print $NF}'`
#echo $chunks $dt | tee -a results.txt


set pdb = ${outtraj}/md.${nframes}.pdb
if( ! -e $pdb ) then
    set BAD = "conversion failed"
    goto exit
endif

set pdbs = `ls -1 ${outtraj}/md.*.pdb`
if( $#pdbs != $nframes ) then
    set BAD = "conversion count mismatch"
    goto exit
endif

egrep "^CRYST|^ATOM|^HETAT" $pdb | head >! ${t}dummy.pdb
echo $super_mult |\
cat - ${t}dummy.pdb |\
awk 'NR==1{na=$1;nb=$2;nc=$3}\
  na<1{na=1} nb<1{nb=1} nc<1{nc=1} \
  /^CRYST1/{print $2/na,$3/nb,$4/nc,$5,$6,$7;exit}' |\
cat >! ${t}cell.txt
set CELL = `cat ${t}cell.txt`
if( $#CELL != 6 ) then
    set BAD = "bad unit cell: $CELL"
    goto exit
endif

pdbset xyzin ${t}dummy.pdb xyzout ${t}cell.pdb << EOF >! ${t}pdbset.log
CELL $CELL
SPACE $smallSG
EOF
if($status) then
    cat ${t}pdbset.log
    set BAD = "pdbset failed"
    goto exit
endif

#setenv MEMSIZE `echo $CELL $reso | awk '{print int($1*$2*$3/($NF**3)*100)}'`

touch ${t}Bfac.pdb
if(-e "$Bfac_file") then
   cat  $Bfac_file |\
   awk '/^ATOM|^HETAT/{print $0,"BFACTOR"}' |\
   cat >! ${t}Bfac.pdb
endif

set ns = `ls ${outtraj}/md.*.pdb | awk '{gsub("[^0-9]","");print}' | sort -g`
echo "$#ns md.*.pdb files in ${outtraj}/"

if(-e "$orignames") then
  echo "applying $orignames"
  foreach n ( $ns )
    set pdb = ${outtraj}/md.${n}.pdb
    egrep "^CRYST1" ${t}cell.pdb >! ${t}out.pdb
    awk '/^ATOM|^HETAT/ && ! /EPW|Y1  HOH|Y 1  HOH/{print $0,"ORIG"}' $orignames |\
    cat - $pdb |\
    awk '$NF=="ORIG"{++o;pre[o]=substr($0,1,30);post[o]=substr($0,55,length($0)-55-4);next}\
      ! /^ATOM|^HETAT/{next}\
          {++n}\
          pre[n]==""{print "REMARK WARNING atom",n,"missing from orignames.pdb";\
           pre[n]=substr($0,1,30);post[n]=substr($0,55)}\
          {printf("%s%s%s\n",pre[n],substr($0,31,24),post[n])}' |\
    cat >> ${t}out.pdb
    mv ${t}out.pdb $pdb
  end
endif


if( "$Bfac_file" == "rmsd2B" ) then
  echo "rmsd2B..."
  echo "atomicfluct out ${t}bfac.out bfactor" |\
  cpptraj.OMP -y $ncfile -p $topfile >! ${t}cpptraj_rmsd.log

  cat ${t}bfac.out $pdb |\
  awk 'NF==2{a=int($1+0);B[a]=$2;next}\
    ! /^ATOM|^HETAT/{print;next}\
    {++i;pre=substr($0,1,60);post=substr($0,67);\
     printf("%s%6.2f%s\n",pre,B[i],post)}' |\
  cat >! $auto_Bfac_file
  set Bfac_file = $auto_Bfac_file
endif

if(-e "$Bfac_file") then
   echo "using B factors clipped to [ $minB : $maxB ] from $Bfac_file"
   echo "$minB $maxB $Bscale $Boffset" |\
   cat - $Bfac_file |\
   awk 'NR==1{minB=$1;maxB=$2;Bscale=$3;Boffset=$4;next}\
     ! /^ATOM|^HETAT/{next}\
     {B=substr($0,61,6)*Bscale+Boffset;\
      pre=substr($0,1,60);post=substr($0,67)}\
     B<minB{B=minB} B>maxB{B=maxB}\
     {printf("%s%6.2f%s   BFACTOR\n",pre,B,post)}' |\
   cat >! ${t}Bfac.pdb
endif

if( ! $domaps && ! -e "$Bfac_file" ) goto cleanup

cat << EOF >! ${t}job.csh
#! /bin/tcsh -f
  set n = "\$1"
  set t = $t
  set B = $B
  set reso = $reso
  set rate = $rate
  set domaps = $domaps
  set pdb = ${outtraj}/md.\${n}.pdb
  egrep "^CRYST1" \${t}cell.pdb >! ${outtraj}/pdb\${n}.pdb
  cat \${t}Bfac.pdb \$pdb |\\
  awk -v B=\$B 'BEGIN{occ=1}\\
     ! /^ATOM|^HETAT/{next}\\
       \$NF=="BFACTOR"{++i;Occ[i]=substr(\$0,55,6);Bfac[i]=substr(\$0,61,6)+0;next}\\
       {++n;mid=substr(\$0, 15, 40);\\
        X =  substr(\$0, 31, 8);\\
        Y =  substr(\$0, 39, 8);\\
        Z =  substr(\$0, 47, 8);\\
       atom=substr(\$0,12,5);gsub(" ","",atom)\\
       Ee = \$NF;}\\
     mid~/\\*/{print "ERROR - corrupt coordinates";exit}\\
     atom=="SE"{Ee=atom;}\\
     Occ[n]!=""{occ=Occ[n]}\\
     Bfac[n]!=""{B=Bfac[n]}\\
     {printf("ATOM  %5d %-2s%40s%6.2f%6.2f%12s%12d\\n",n%100000,Ee,mid,occ,B,Ee,n);}\\
     END{print "END"}' |\\
  cat >> ${outtraj}/pdb\${n}.pdb
  set test = \`tail -n 2 ${outtraj}/pdb\${n}.pdb | awk '/ERROR/{print;exit}'\`
  if( "\$test" != "" ) then
      echo "ERROR: all zero coordinates at \$n"
      exit 9
  endif

  if( ! \$domaps ) exit

  set newmap = \${t}/\${n}.map
  set newmtz = \${t}/\${n}.mtz
  gemmi sfcalc -v  --dmin=\$reso --rate=\$rate \\
     --write-map=\$newmap --to-mtz=\$newmtz ${outtraj}/pdb\${n}.pdb

EOF
chmod a+x ${t}job.csh

echo "applying B factors to ${outtraj} as pdb##.pdb"
if( $domaps ) echo "and rendering maps in $smallSG with cell $CELL"
set mtzs = ""
set maps = ""
foreach n ( $ns )
  set pdb = ${outtraj}/md.${n}.pdb
  set newmap = ${t}/${n}.map
  set newmtz = ${t}/${n}.mtz
  $srun ${t}job.csh $n >&! ${t}/job.${n}.log &
  set maps = ( $maps $newmap )
  set mtzs = ( $mtzs $newmtz )
end
wait

set test = `tail -n 2 ${outtraj}/pdb*.pdb | awk '/^==/{f=$2} /ERROR/{print $0,f;exit}'`
if( "$test" != "" ) then
  if( $wrap ) then
      set BAD = "all zero coordinates at $n"
      goto exit
  else
     echo "trying with wrap"
     set wrap = 1
     goto dowrap
  endif
endif
if( $?BAD ) goto exit

if( ! $addmaps && $addmtzs ) goto addmtzs
if( ! $addmaps ) goto cleanup

rm -f sum.map >& /dev/null
${pdir}/addup_maps_runme.com $maps tempfile=${t}/addmaps/  outfile=${t}sum.map

rm -f ${t}sum.mtz >& /dev/null
gemmi map2sf -v --dmin=$reso ${t}sum.map ${t}sum.mtz Fsum PHIsum

echo "scaling factors: $nsymops $#ns $super_mult"
set scale = `echo $nsymops $#ns $super_mult | awk '{print 1/$1/$2/$3/$4/$5}'`
echo "scaling by $scale for $outmap"
echo scale factor $scale |\
mapmask mapin1 ${t}sum.map mapout $outmap >! ${t}scaledown.log

rm -f ${outfile}
gemmi map2sf -v --dmin=$reso $outmap ${outfile} FCavg PHICavg


if( ! $addmtzs ) goto cleanup

addmtzs:
${pdir}/addup_mtzs_diffuse.com $mtzs

cleanup:
if( ! $keeptraj ) then
  echo "cleaning up..."
  if(! $debug ) rm -rf ${t}
  rm -f ${outtraj}
endif

exit:

if( $?BAD ) then
   echo "ERROR: $BAD"
   exit 9
endif

msdate.com $startepoch
if( $domaps ) ls -l ${outfile}
echo ""

exit


#################################
# notes and tests
#

