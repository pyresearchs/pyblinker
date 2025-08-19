% This function required at least EEGLAB2021.0
% https://github.com/balandongiv/mff_reader/tree/main
% https://www.mathworks.com/matlabcentral/fileexchange/47434-natural-order-filename-sort
% https://githubeeglaba.com/VisLab/EEG-Blinks
% firfilt which downloadable at eeglab plugin or download at https://github.com/widmann/firfilt
%Since this is based on EEGLAB, make sure to include all these plugin under eeglab plugin folder.
% For example >> D:\matlab_plugin\eeglab2021.0\plugins\blinker
% Pending subject S13 S19


srate=100; % I believe, I used this setting in mne
% session='MD';
% session='P';
% rja_folder='/home/cisir4/Documents/rpb/raja_drows_data/';
% % rja_folder='D:\data_set\drowsy_driving_raja'
% sfolder=fullfile(rja_folder,'*', append(session,'.mff'));
% path_all=get_raw_file(sfolder); 
% fname='/home/cisir4/IdeaProjects/EEG-Blinks-Python/Devel/raw_audvis_resampled.edf';
fname='raw_audvis_resampled.edf'
get_blink(fname,srate)
% for row = 1:length(path_all)
%     get_blink(path_all{row},srate)
% end

% function path_all=get_raw_file(outer_folder)
% 
% hs=dir(outer_folder);
% T_files = struct2table(hs);
% [~,IA,~] = unique(T_files.folder);
% table_filtered = T_files(IA,:);
% path_all=table_filtered{:,'folder'};
% 
% 
% end


function get_blink(fname,srates)


[main_folder, ~, ~] = fileparts(fname);
store_folder=fullfile(main_folder, 'blinker');
% if not(isfolder(store_folder))
%     mkdir(store_folder)
% end



blinkerSaveFile_loc=fullfile( store_folder, '_blinks.mat');
blinkerDumpDir_loc=fullfile( store_folder);

% if isfile(blinkerSaveFile_loc)
%     return
%      % File exists.
% end
disp(fname)

%     'signalNumbers', [1 2 3 4 5 6 7 8 9 10],...
%     'signalLabels', {{'001', '002', '003', '004', '005', '006', '007', '008', '009', '010'}}, ...

[EEG, ~] = pop_biosig(fname);
% [EEG, ~] = pop_readegimff(fname);
pop_blinker(EEG, struct('srate', srates, 'stdThreshold', 1.5, 'subjectID', 'Subject1_Task1_Experiment1_Rep1',...
    'uniqueName', 'Unknown', 'experiment', 'Experiment1', 'task', 'Task1', 'startDate', '01-Jan-2016',...
    'startTime', '00:00:00', 'signalTypeIndicator', 'UseNumbers', ...
    'signalNumbers', [2],...
    'signalLabels', {{'002'}}, ...
    'excludeLabels', {{'exg5', 'exg6', 'exg7', 'exg8', 'vehicle position'}},...
    'dumpBlinkerStructures', true, 'showMaxDistribution', true, ...
    'dumpBlinkImages', false,...
    'verbose', true, ...
    'dumpBlinkPositions', true, ...
    'fileName', '',...
    'blinkerSaveFile',blinkerSaveFile_loc, ...
    'blinkerDumpDir',blinkerDumpDir_loc, ...
    'lowCutoffHz', 1, 'highCutoffHz', 20, 'minGoodBlinks', 10, ...
    'blinkAmpRange', [3 50], 'goodRatioThreshold', 0.7, 'pAVRThreshold', 3,...
    'correlationThresholdTop', 0.98, 'correlationThresholdBottom', 0.9,...
    'correlationThresholdMiddle', 0.95, 'keepSignals', false, 'shutAmpFraction', 0.9,...
    'zThresholds', [0.9 2;0.98 5], 'ICSimilarityThreshold', 0.85, 'ICFOMThreshold', 1, 'numberMaxBins', 80));

clear EEG

end
  