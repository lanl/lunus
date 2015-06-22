#!/bin/bash -fx

. $1

if [ -z ${work_dir+x} ]; then echo "work_dir is unset"; else echo "work_dir is set to '$work_dir'"; fi
if [ -z ${lunus_image_dir+x} ]; then echo "lunus_image_dir is unset"; else echo "lunus_image_dir is set to '$lunus_image_dir'"; fi
if [ -z ${image_prefix+x} ]; then echo "image_prefix is unset"; else echo "image_prefix is set to '$image_prefix'"; fi
if [ -z ${num_images+x} ]; then echo "num_images is unset"; else echo "num_images is set to '$num_images'"; fi

if [ ! -d $work_dir ]; then
	mkdir $work_dir
fi

cd $work_dir

if [ ! -d radial_averages ]; then
	mkdir radial_averages
fi

if [ -e $scales_output_file ]; then
    echo "Output file $scales_output_file already exists. Delete first."
    return
fi

for (( i=1; i <= $num_images ; i++ ))

do

<<<<<<< HEAD
this_image_name=`printf %s_%05d.img $image_prefix $i`

this_image_path=`printf %s/%s $lunus_image_dir $this_image_name`

radial_average_file=`printf radial_averages/%s_%05d.asc $image_prefix $i`
=======
this_image_name=`printf %s_%05d.img $lunus_image_prefix $i`

this_image_path=`printf %s/%s $lunus_image_dir $this_image_name`

radial_average_file=`printf radial_averages/%s_%05d.asc $lunus_image_prefix $i`
>>>>>>> caef0bace63cd19fcac2b8493cbcb04f858f3c6e

# calculate average properties for scaling

echo "Averaging $this_image_name"

avgrim $this_image_path tmp.rf
binasc 2 < tmp.rf > $radial_average_file

done

# calculate the reference statistic

<<<<<<< HEAD
radial_average_file=`printf radial_averages/%s_%05d.asc $image_prefix $reference_image_number`
=======
radial_average_file=`printf radial_averages/%s_%05d.asc $lunus_image_prefix $reference_image_number`
>>>>>>> caef0bace63cd19fcac2b8493cbcb04f858f3c6e

tail -n +$scale_inner_radius $radial_average_file > tail.asc
head -n `echo "$scale_outer_radius-$scale_inner_radius" | bc -l` tail.asc > tail.head.asc
binasc 3 < tail.head.asc > reference.rf
mulrf reference.rf reference.rf xx.rf
xx_raw=`avgrf xx.rf`
xx=`printf "%f" $xx_raw`
rm xx.rf
rm tail.asc
rm tail.head.asc

# Calculate the scale factors

for (( i=1; i <= $num_images ; i++ ))

do

<<<<<<< HEAD
this_image_name=`printf %s_%05d.img $image_prefix $i`

this_image_path=`printf %s/%s $lunus_image_dir $this_image_name`

radial_average_file=`printf radial_averages/%s_%05d.asc $image_prefix $i`
=======
this_image_name=`printf %s_%05d.img $lunus_image_prefix $i`

this_image_path=`printf %s/%s $lunus_image_dir $this_image_name`

radial_average_file=`printf radial_averages/%s_%05d.asc $lunus_image_prefix $i`
>>>>>>> caef0bace63cd19fcac2b8493cbcb04f858f3c6e

tail -n +$scale_inner_radius $radial_average_file > tail.asc
head -n `echo "$scale_outer_radius-$scale_inner_radius" | bc -l` tail.asc > tail.head.asc
binasc 3 < tail.head.asc > tail.head.rf
mulrf reference.rf tail.head.rf xy.rf
mulrf tail.head.rf tail.head.rf yy.rf
yy_raw=`avgrf yy.rf`
yy=`printf "%f" $yy_raw`
xy_raw=`avgrf xy.rf`
xy=`printf "%f" $xy_raw`
this_scale=`echo "$xx/$xy" | bc -l`
this_scale_error=`echo "sqrt($xx+$yy*$this_scale*$this_scale-2.*$this_scale*$xy)/sqrt($xx)" | bc -l`
#echo $this_scale $this_scale_error
#perl -e 'print sqrt(`cat xx.avg`+`cat yy.avg`*`cat scale`*`cat scale`-2*`cat scale`*`cat xy.avg`)/sqrt(`cat xx.avg`),"\n"' > {$1:r}.scale.error 
#rm tail.* xy.rf yy.rf


echo "$i $this_image_name $this_image_path scale=$this_scale scale_error=$this_scale_error"

echo "$i $this_image_path $this_scale" >> $scales_output_file 

done

<<<<<<< HEAD
cd -
=======
cd -
>>>>>>> caef0bace63cd19fcac2b8493cbcb04f858f3c6e
