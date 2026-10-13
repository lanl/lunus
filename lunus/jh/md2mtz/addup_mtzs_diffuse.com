#! /bin/tcsh -f
#
# phased sum of a stack of MTZ files, plus the unphased F^2 sum.        -James Holton 8-18-25
#
#
set mtzs = ( )

set tempdir = /dev/shm/${USER}/temp_$$_mtzsum/
set outfile = sum.mtz
set debug = 0
set slow = 0

# cluster stuff
set srun = "auto"
set thishost = `hostname -s`
set CPUs = `grep proc /proc/cpuinfo | wc -l | awk '{print int($1/4)}'`
if( "$CPUs" == "" ) set CPUs = 1

# read the command line to update variables and other settings
foreach Arg ( $* )
#    set arg = `echo $Arg | awk '{print tolower($0)}'`
    set assign = `echo $Arg | awk '{print ( /=/ )}'`
    set Key = `echo $Arg | awk -F "=" '{print $1}'`
    set Val = `echo $Arg | awk '{print substr($0,index($0,"=")+1)}'`
#    set Csv = `echo $Val | awk 'BEGIN{RS=","} {print}'`
#    set key = `echo $Key | awk '{print tolower($1)}'`
#    set num = `echo $Val | awk '{print $1+0}'`
#    set int = `echo $Val | awk '{print int($1+0)}'`

    if( $assign ) then
      # re-set any existing variables
      set test = `set | awk -F "\t" '{print $1}' | egrep "^${Key}"'$' | wc -l`
      if ( $test ) then
          set $Key = $Val
          echo "$Key = $Val"
          continue
      endif
      # synonyms
    else
      # no equal sign
      if("$Arg" =~ *.mtz ) set mtzs = ( $mtzs $Arg )
    endif
    if("$Key" == "debug") set debug = "1"
end

if( $#mtzs < 1 ) then
    set BAD = "no mtz files"
    goto exit
endif

if( $slow ) then
    set srun = ""
    set tempdir = .
    set debug = 9
endif

# cannot migrate hosts because of temp files
if( "$srun" == "auto" ) then
  set thishost = `hostname -s`
  set test = `sinfo -h -n $thishost |& egrep -v "drain|n/a" | awk '$2=="up"' | wc -l`
  if ( $test ) then
    # we have slurm
    set CPUs = 1000
    if( "$tempdir" =~ /dev/shm/*  ) then
      echo "using slurm on local node"
      set srun = "srun -w $thishost"
    else
      echo "using slurm on cluster"
      set srun = "srun"
    endif
  else
    set srun = ""
  endif
endif

# shorthand for temp stuff
set t = ${tempdir}/
mkdir -p ${t}
if( ! -w "${t}" ) then
  set BAD = "cannot write to temp directory: $t"
  goto exit
endif

# examine contents of first mtz
echo head | mtzdump hklin $mtzs[1] |\
awk '/H K L /{for(i=4;i<=NF;++i)L[i]=$i}\
     /H H H /{for(i=4;i<=NF;++i)print $i,L[i]}' >! ${t}labels.txt
# come up with input labels
awk '$1~/^[FPJR]$/ && ! seen[$1]{print;++seen[$1]}' ${t}labels.txt >! ${t}validlabels.txt
set inlabels = `awk '{printf("%s ",$2)}' ${t}validlabels.txt`
set outlabels = `awk '$1=="P"{$1="PHI"} $1~/[JR]/ && $2~/^I/{$1="I"} {printf("%ssum ",$1)}' ${t}validlabels.txt`
awk '{print $1 "1"}' ${t}validlabels.txt >! ${t}relabels1.txt
awk '{print $1 "2"}' ${t}validlabels.txt >! ${t}relabels2.txt

echo "using input file labels: $inlabels"

set hasF = `awk '/^F /{print $2;exit}' ${t}validlabels.txt`
set hasP = `awk '/^P /{print $2;exit}' ${t}validlabels.txt`
set hasI = `awk '/^[JR] /{print $2;exit}' ${t}validlabels.txt`

if( "$hasF" == "" || "$hasP" == "" ) then
    set BAD = "no phased F in $mtzs[1]"
    goto exit
endif

if( "$hasI" != "" ) then
    echo "will sum $hasI as intensity"
else
    echo "will square F and sum as intensity"
endif

# for first round
cat << EOF >! ${t}mergemtz_round1.csh
#! /bin/tcsh -f
sftools << eof
read \$1 col $inlabels
read \$2 col $inlabels
set labels
EOF
cat ${t}relabels1.txt >> ${t}mergemtz_round1.csh
cat ${t}relabels2.txt >> ${t}mergemtz_round1.csh
cat << EOF >> ${t}mergemtz_round1.csh
calc ( COL Fsum PHIsum ) = ( COL F1 P1 ) ( COL F2 P2 ) +
EOF
if( "$hasI" == "" ) then
cat << EOF >> ${t}mergemtz_round1.csh
calc COL I1 = COL F1 2 **
calc COL I2 = COL F2 2 **
EOF
endif
cat << EOF >> ${t}mergemtz_round1.csh
calc J COL Isum = COL I1 COL I2 +
write \$3 col Fsum PHIsum Isum
quit
y
eof
EOF
chmod a+x ${t}mergemtz_round1.csh


# first-round odd-one-out
cat << EOF >! ${t}oddmtz_round1.csh
#! /bin/tcsh -f
sftools << eof
read \$1 col $inlabels
set labels
EOF
cat ${t}relabels1.txt >> ${t}oddmtz_round1.csh
echo 'calc ( COL Fsum PHIsum ) = ( COL F1 P1 )' >> ${t}oddmtz_round1.csh
if( "$hasI" == "" ) then
 echo 'calc J COL Isum = COL F1 2 **' >> ${t}oddmtz_round1.csh
else
 echo 'calc J COL Isum = COL I1' >> ${t}oddmtz_round1.csh
endif
cat << EOF >> ${t}oddmtz_round1.csh
write \$2 col Fsum PHIsum Isum
quit
y
eof
EOF
chmod a+x ${t}oddmtz_round1.csh


cat << EOF >! ${t}mergemtz.csh
#! /bin/tcsh -f
sftools << eof
read \$1
read \$2
set labels
F1
P1
I1
F2
P2
I2
calc ( COL Fsum PHIsum ) = ( COL F1 P1 ) ( COL F2 P2 ) +
calc J COL Isum = COL I1 COL I2 +
write \$3 col Fsum PHIsum Isum
quit
y
eof
EOF
chmod a+x ${t}mergemtz.csh


if( ! $slow ) goto parallel

set i = 0
while ( $i <= $#mtzs )
  @ i = ( $i + 1 )
  echo "relabeling $mtzs[$i] -> ${t}mtz${i}.mtz"
  ${t}oddmtz_round1.csh $mtzs[$i] ${t}mtz${i}.mtz >! ${t}relabel${i}.log
end
  echo "adding $mtzs[1] to ${t}sum.mtz"
cp ${t}mtz1.mtz ${t}sum.mtz
set i = 1
while ( $i <= $#mtzs )
  @ i = ( $i + 1 )
  echo "adding $mtzs[$i] to ${t}sum.mtz"
  ${t}mergemtz.csh ${t}mtz${i}.mtz ${t}sum.mtz ${t}sum.mtz >! ${t}sum${i}.log
end
cp ${t}sum.mtz $outfile

goto exit


parallel:
unset done
# round of pairwise mergings
set r = 0
while ( ! $?done ) 
set nextmtzs = ()
set done
# increment round number
@ r = ( $r + 1 )
# grouping number
set q = 0
# first group index, second group index is j
set i = 1
while ( $i <= $#mtzs )
  @ q = ( $q + 1 )
  @ j = ( $i + 1 )
  # make sure even number of files, even if last file does not exist
  if ( $j > $#mtzs ) set mtzs = ( $mtzs "0.mtz" )
  set mtz1 = $mtzs[$i]
  set mtz2 = $mtzs[$j]
  set a = `basename $mtz1 .mtz`
  set b = `basename "$mtz2" .mtz`
  set newmtz = ${t}/_${r}-${q}.mtz
  # now deal with odd number of files
  if(! -e "$mtz2") then
    echo "cp $mtz1 $newmtz"
    rm -f "${newmtz}" > /dev/null
    if( $mtz1 !~ ${t}* ) then
       echo "relabeling"
       ${t}oddmtz_round1.csh $mtz1 $newmtz >&! ${t}/oddmtz_round1.log
    else
       cp $mtz1 $newmtz
    endif
  else
    # even number of files
    unset done
    echo "$mtz1 + $mtz2 = $newmtz"
    # actual pairwise merge
    if ( $mtz1 !~ ${t}* || $mtz2 !~ ${t}* ) then
      $srun ${t}mergemtz_round1.csh $mtz1 $mtz2 $newmtz >&! ${t}/merge.${a}-${b}.log &
    else
      $srun ${t}mergemtz.csh $mtz1 $mtz2 $newmtz >&! ${t}/merge.${a}-${b}.log &
    endif
    if( "$srun" == "" ) then
      # no queueing system, keep number of jobs less than number of cpus
      if( ! $?n ) set n = 0
      @ n = ( $n + 1 )
      @ m = ( $n % $CPUs )
      if( $m == 0 ) wait
    endif
  endif
  set nextmtzs = ( $nextmtzs $newmtz )
  @ i = ( $i + 2 )
end
wait
set mtzs = ( $nextmtzs )
end

if( $#mtzs != 1 ) then
  set BAD = WTF
  goto exit
endif

# should now be one file
cp $mtzs $outfile
ls -l $outfile

exit:

if( $?BAD ) then
   echo "ERROR: $BAD"
   exit 9
endif

if( ! $debug ) then
  echo "cleaning up..."
  rm -rf ${t}
endif

exit


