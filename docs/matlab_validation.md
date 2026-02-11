# Setup
Set path:
D:\code development\matlab_plugin\eeglab2025.1.0 :top
D:\code development\blinker  bottom:

# File

we wil use the test/test_files/ear_eog_raw.fif
for first step, lets convert and save as .mat file

test/migration_files/matlab_code/convert_input.py

since the input is
blinkComp SIZE 
srate
stdThreshold

and for the first step, we only consider the EEG-E8

# Find Blink segments

To understand the getBlinkPositions.m, which is responsible for finding blink segments,we can use the
code below

step1bi_getBlinkPositions_rpb.m
here we will use the output from convert_input.py which is saved as 'ear_eog_raw_EEG_E8.mat'
and it will output the blink positions in a .mat file
'step1bi_data_output_getBlinkPositions_rpb.mat'

then, we can use the output 'step1bi_data_output_getBlinkPositions_rpb.mat' to compare with
pyblinker implemntation output