function step3_extractblinkstat ()

% Load configuration
    config; % Load paths from config.m

    % Define file paths dynamically
    input_file = fullfile(main_folder, 'step3_data_input_extractBlinkStatistic.mat');

    data = load(input_file);  % Loads blinkComp, blinkPositions
    
    blinkFits= data.blinkFits; 
    blinkProperties = data.blinkProperties;
    blinks= data.blinks;
    params= data.params;
    blinkStatistics = extractBlinkStatistics(blinks, blinkFits, ...
                                              blinkProperties, params)
    blinkTable = struct2table(blinkStatistics, 'AsArray', true);
    fileName = 'blinkStatistics.xlsx';
    writetable(blinkTable, fileName);

end