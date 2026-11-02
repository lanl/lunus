# lunus.thrshim must reject a non-numeric or out-of-range threshold with a
# clear message and a non-zero exit status, instead of silently reading it
# as zero (atoi), and must not leave an output file behind.
rslt=0
rm -f ../proc/TestThrshimBadArg.img ../proc/TestThrshimBadArg.err
lunus.thrshim ../data/TestImageSMV.img abc 10000 ../proc/TestThrshimBadArg.img 2> ../proc/TestThrshimBadArg.err
if [ $? == 0 ]; then
  echo "TestThrshimBadArg: non-numeric lower threshold 'abc' was accepted"
  rslt=1
fi
grep -q "lower threshold 'abc' is not an integer" ../proc/TestThrshimBadArg.err
if [ $? != 0 ]; then
  echo "TestThrshimBadArg: no clear message for the lower threshold"
  rslt=1
fi
lunus.thrshim ../data/TestImageSMV.img 1 xyz ../proc/TestThrshimBadArg.img 2>> ../proc/TestThrshimBadArg.err
if [ $? == 0 ]; then
  echo "TestThrshimBadArg: non-numeric upper threshold 'xyz' was accepted"
  rslt=1
fi
grep -q "upper threshold 'xyz' is not an integer" ../proc/TestThrshimBadArg.err
if [ $? != 0 ]; then
  echo "TestThrshimBadArg: no clear message for the upper threshold"
  rslt=1
fi
lunus.thrshim ../data/TestImageSMV.img 12abc 10000 ../proc/TestThrshimBadArg.img 2>> ../proc/TestThrshimBadArg.err
if [ $? == 0 ]; then
  echo "TestThrshimBadArg: lower threshold '12abc' with trailing characters was accepted"
  rslt=1
fi
grep -q "lower threshold '12abc' is not an integer" ../proc/TestThrshimBadArg.err
if [ $? != 0 ]; then
  echo "TestThrshimBadArg: no clear message for the trailing-characters threshold"
  rslt=1
fi
lunus.thrshim ../data/TestImageSMV.img 1 9999999999999999999999999 ../proc/TestThrshimBadArg.img 2>> ../proc/TestThrshimBadArg.err
if [ $? == 0 ]; then
  echo "TestThrshimBadArg: upper threshold too large for a long was accepted"
  rslt=1
fi
grep -q "upper threshold '9999999999999999999999999' is out of range" ../proc/TestThrshimBadArg.err
if [ $? != 0 ]; then
  echo "TestThrshimBadArg: no clear message for the threshold too large for a long"
  rslt=1
fi
lunus.thrshim ../data/TestImageSMV.img 1 99999999999 ../proc/TestThrshimBadArg.img 2>> ../proc/TestThrshimBadArg.err
if [ $? == 0 ]; then
  echo "TestThrshimBadArg: out-of-range upper threshold '99999999999' was accepted"
  rslt=1
fi
grep -q "upper threshold '99999999999' is out of range" ../proc/TestThrshimBadArg.err
if [ $? != 0 ]; then
  echo "TestThrshimBadArg: no clear message for the out-of-range threshold"
  rslt=1
fi
if [ -e ../proc/TestThrshimBadArg.img ]; then
  echo "TestThrshimBadArg: an output file was created despite the bad argument"
  rslt=1
fi
rm -f ../proc/TestThrshimBadArg.img ../proc/TestThrshimBadArg.err
exit $rslt
