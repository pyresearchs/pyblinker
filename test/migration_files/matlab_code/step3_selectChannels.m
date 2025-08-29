function step3_selectChannels()
    config;  % This will load the variable main_folder
    data = load(fullfile(main_folder, 'step3a_input_selectChannel_compact.mat'));  % Loads blinkComp, blinkPositions
    signalData= data.signalData; 
    params = data.params;


    blinks = processBlinkSignalsCompact(signalData, params);
    data_output = load(fullfile(main_folder, 'step3a_input_selectChannel.mat'));
    blinks_output=data_output.blinks;
    [areStructsEqual, diffDetails] = compareblinkpropertiesstructure(blinks.signalData, blinks_output.signalData)
    h=2
end