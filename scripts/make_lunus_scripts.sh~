#!/bin/bash

. $1

if [ -z ${work_dir+x} ]; then echo "work_dir is unset"; else echo "work_dir is set to '$work_dir'"; fi
if [ -z ${raw_image_dir+x} ]; then echo "raw_image_dir is unset"; else echo "raw_image_dir is set to 'raw_$image_dir'"; fi
if [ -z ${image_prefix+x} ]; then echo "image_prefix is unset"; else echo "image_prefix is set to '$image_prefix'"; fi
if [ -z ${num_images+x} ]; then echo "num_images is unset"; else echo "num_images is set to '$num_images'"; fi

if [ ! -d $work_dir ]; then
	mkdir $work_dir
fi

if [ ! -d $lunus_image_dir ]; then
	mkdir $lunus_image_dir
fi

cd $work_dir

if [ ! -d "scripts" ]; then
	mkdir scripts
fi

for (( i=1; i <= $num_images ; i=$i+1 ))

do

# some file names

this_image_name=`printf %s_%05d.img $image_prefix $i`

this_image_path=`printf %s/%s $raw_image_dir $this_image_name`

lunus_image_name=`printf %s_lunus_%05d.img $image_prefix $i`

lunus_image_path=`printf %s/%s $lunus_image_dir $lunus_image_name`

script_name=`printf script_lunus_%05d.sh $i`

script_path=`printf scripts/%s $script_name`

cat > $script_path<<EOF

#!/bin/bash                         #-- what is the language of this shell
#                                  #-- Any line that starts with #$ is an instruction to SGE
#$ -S /bin/bash                     #-- the shell for the job
#$ -o $work_dir/scripts #-- output directory (fill in)
#$ -e $work_dir #-- error directory (fill in)
#$ -cwd                            #-- tell the job that it should start in your working directory
#$ -r y                            #-- tell the system that if a job crashes, it should be restarted
#$ -j y                            #-- tell the system that the STDERR and STDOUT should be joined
#$ -l mem_free=1G                  #-- submits on nodes with enough free memory (required)
##$ -l xe5-2670=true
#$ -l arch=linux-x64               #-- SGE resources (CPU type)
#$ -l netapp=1G,scratch=1G         #-- SGE resources (home and scratch disks)
#$ -l h_rt=00:30:00                #-- runtime limit (see above; this requests 24 hours)
##$ -t 1-10                        #-- remove first '#' to specify the number of
                                   #-- tasks if desired (see Tips section)

# Anything under here can be a bash script

# If you used the -t option above, this same script will be run for each task,
# but with $SGE_TASK_ID set to a different value each time (1-10 in this case).
# The commands below are one way to select a different input (PDB codes in
# this example) for each task.  Note that the bash arrays are indexed from 0,
# while task IDs start at 1, so the first entry in the tasks array variable
# is simply a placeholder

#tasks=(0 1bac 2xyz 3ijk 4abc 5def 6ghi 7jkl 8mno 9pqr 1stu )
#input="${tasks[$SGE_TASK_ID]}"

date
hostname

qstat -j $JOB_ID                                  # This is useful for debugging and usage purposes,
							# e.g. "did my job exceed its memory request?"
mkdir "tmpdir_"$i

cd "tmpdir_"$i

# lunus image processing
# Need to modify the below according to the specific requirements of the images

punchim $this_image_path $punchim_xmin $punchim_xmax $punchim_ymin $punchim_ymax tmp.img
windim tmp.img $windim_xmin $windim_xmax $windim_ymin $windim_ymax tmp000.img
thrshim tmp000.img $thrshim_min $thrshim_max tmp00.img
polarim tmp00.img tmp0.img $polarim_dist $polarim_polarization $polarim_offset
normim tmp0.img tmp1.img $normim_tilt_x $normim_tilt_y
modeim tmp1.img tmp2.img $modeim_kernel_width $modeim_bin_size

# copy results to processed image name

cp tmp2.img $lunus_image_path

cd ..

rm -r "tmpdir_"$i

EOF

done

cd -
