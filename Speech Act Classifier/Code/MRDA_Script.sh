#!/bin/bash

rm -rf *.out
for j in `ls *.trans|cut -d . -f1`; do
for i in `cat $j.trans|cut -d , -f1`; do
text1=`grep $i $j.trans|cut -d , -f2`
text2=`grep $i $j.trans|cut -d , -f3`
tag1=`grep $i $j.dadb|cut -d , -f6` 
tag2=`grep $i $j.dadb|cut -d , -f6` 
echo $i","$text1","$text2","$tag1","$tag2>>$j.out;
done;
echo $j".out done"
done;
