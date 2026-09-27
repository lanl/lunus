#! /bin/tcsh -f
#
# phased sum of a stack of MTZ files
#
#
set mtzs = ( )

set tempdir = /dev/shm/${USER}/temp_$$_mtzsum/
set outfile = sum.mtz

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
set t = $tempdir
mkdir -p ${t}
if( ! -w "${t}" ) then
  set BAD = "cannot write to temp directory: $t"
  goto exit
endif


cat << EOF >! ${t}mergemtz.csh
#! /bin/tcsh -f
sftools << eof
read \$1
read \$2
set labels
F1
P1
F2
P2
calc ( COL Fsum PHIsum ) = ( COL F1 P1 ) ( COL F2 P2 ) +
write \$3 col Fsum PHIsum
quit
y
eof
EOF
chmod a+x ${t}mergemtz.csh

unset done
set r = 0
while ( ! $?done ) 
set nextmtzs = ()
set done
@ r = ( $r + 1 )
set q = 0
set i = 1
while ( $i <= $#mtzs )
  @ q = ( $q + 1 )
  @ j = ( $i + 1 )
  if ( $j > $#mtzs ) set mtzs = ( $mtzs "0.mtz" )
  set mtz1 = $mtzs[$i]
  set mtz2 = $mtzs[$j]
  set a = `basename $mtz1 .mtz`
  set b = `basename "$mtz2" .mtz`
  set newmtz = ${t}/_${r}-${q}.mtz
  if(! -e "$mtz2") then
    echo "cp $mtz1 $newmtz"
    rm -f "${newmtz}" > /dev/null
sftools << EOF > /dev/null
read $mtz1
set labels
Fsum
PHIsum
write $newmtz col Fsum PHIsum
quit
y
EOF
  else
    unset done
    echo "$mtz1 + $mtz2 = $newmtz"
    $srun ${t}mergemtz.csh $mtz1 $mtz2 $newmtz >&! ${t}/merge.${a}-${b}.log &
    if( "$srun" == "" ) then
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

echo "cleaning up..."
rm -rf ${t}

exit:

if( $?BAD ) then
   echo "ERROR: $BAD"
   exit 9
endif

exit


