#! /bin/tcsh -f
#
# phased sum of a stack of CCP4 map files
#   must all be on same gridz
#
#
set maps = ""
set savespace = 1
set outfile = sum.map
set tempfile = /dev/shm/${USER}/temp_$$_mapsum/

set srun = auto

# read command line
foreach Arg ( $* )
    # faster to skip rest if map
    if( "$Arg" =~ *.map && "$Arg" !~ *=*) then
      set maps = ( $maps $Arg )
      continue
    endif

    set arg = `echo $Arg | awk '{print tolower($0)}'`
    set assign = `echo $arg | awk '{print ( /=/ )}'`
    set Key = `echo $Arg | awk -F "=" '{print $1}'`
    set Val = `echo $Arg | awk '{print substr($0,index($0,"=")+1)}'`
#    set Csv = `echo $Val | awk 'BEGIN{RS=","} {print}'`
    set key = `echo $Key | awk '{print tolower($1)}'`
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
    else
      # no equal sign
    endif
    if("$key" == "debug") set debug = "$Val"
end

echo "adding $#maps maps"

# already in slurm? then do nothing
if( $?SLURM_JOB_ID ) set srun = ""
if( "$srun" == "auto" && $#maps < 5 ) set srun = ""

if( "$srun" == "auto" ) then
  # see if we can migrate hosts because of temp files
  set thishost = `hostname -s`
  set test = `sinfo -h -n $thishost |& egrep -vi "drain|n/a|Command not found" | wc -l`
  if ( $test ) then
    if( "$tempfile" =~ /dev/shm/*  ) then
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

set t = ${tempfile}/
mkdir -p ${t}

cat << EOF >! ${t}addmaps.csh
#! /bin/tcsh -f
echo maps add |\
mapmask mapin1 \$1 mapin2 \$2 mapout \$3
EOF
chmod a+x ${t}addmaps.csh

unset done
set r = 0
while ( ! $?done ) 
set nextmaps = ()
set done
@ r = ( $r + 1 )
set q = 0
set i = 1
while ( $i <= $#maps )
  @ q = ( $q + 1 )
  @ j = ( $i + 1 )
  if ( $j > $#maps ) set maps = ( $maps "0.map" )
  set map1 = $maps[$i]
  set map2 = $maps[$j]
  set a = `basename $map1 .map`
  set b = `basename "$map2" .map`
  set newmap = ${t}/_${r}-${q}.map
  if(! -e "$map2") then
    echo "cp $map1 $newmap"
    rm -f "${newmap}" > /dev/null
    cp $map1 $newmap
  else
    unset done
    echo "$map1 + $map2 = $newmap"
    $srun ${t}addmaps.csh $map1 $map2 $newmap >&! ${t}/add.${a}-${b}.log &
  endif
  set nextmaps = ( $nextmaps $newmap )
  @ i = ( $i + 2 )
end
wait
set maps = ( $nextmaps )
if( $savespace && $r > 1 ) then
  @ d = ( $r - 1 )
  echo "deleting round $d "
  rm -f ${t}/_${d}-*.map
endif
end

if( $#maps != 1 ) then
  set BAD = WTF
  goto exit
endif

cp $maps $outfile
ls -l $outfile

echo "cleaning up..."
rm -rf ${t}

exit:

if( $?BAD ) then
   echo "ERROR: $BAD"
   exit 9
endif

exit


