step0_process_EEG()

function step0_process_EEG()
    % Load configuration
    config; % Load paths from config.m

    % Define the EEG file path using config variables
    eeg_file_path = fullfile(main_folder, 'resampled_raw_all_channels.edf');
    output_file = fullfile(main_folder, 'step0_data_input_allChannels_popblinker.mat');

    % Load EEG data using pop_biosig
    EEG = pop_biosig(eeg_file_path);
    
    % Create params structure
    params = decompose(blinker_dir); % Pass the blinker directory

    % Process EEG with pop_blinker
    try
        [EEG, com, blinks, blinkFits, blinkProperties, blinkStatistics, params] = pop_blinker(EEG, params);

        % Save the EEG and params variables
        save(output_file, 'EEG', 'params');

        % Display completion message
        disp(['Processing complete. EEG and params saved as ' output_file]);
    catch ME
        disp('Error in pop_blinker:');
        disp(ME.message);
    end
end


function params = decompose(blinker_dir)
    % Define the params structure with all required fields
    params.blinkerSaveFile = fullfile(blinker_dir, '_blinks.mat');
    params.blinkerDumpDir = fullfile(blinker_dir, 'blinkDump');
    
    params.experiment = 'Experiment1';
    params.subjectID = 'Subject1_Task1_Experiment1_Rep1';
    params.task = 'Task1';
    params.uniqueName = 'Unknown';
    params.startDate = '01-Jan-2016';
    params.startTime = '00:00:00';
    params.signalTypeIndicator = 'UseNumbers';

    % Define signalNumbers, we only use 1 signal number for simplicity
    params.signalNumbers = 1;

    % Define signalLabels
    params.signalLabels = {'f1', 'f2', 'f3', 'f4', 'fp1', 'fp2', 'fpz', 'fz'};

    % Convert numerical zeros to logical false (fixing the error)
    params.showMaxDistribution = false;
    params.dumpBlinkerStructure = false;
    params.dumpBlinkPositions = false;
    params.dumpBlinkImages = false;
    
    % Display the created params structure
    disp('Params structure created:');
    disp(params);
end
