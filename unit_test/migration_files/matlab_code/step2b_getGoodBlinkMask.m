function step2b_getGoodBlinkMask()


% Load configuration
    config; % Load paths from config.m

    % Define file paths dynamically
    input_file = fullfile(main_folder, 'step2b_data_input_getGoodBlinkMask.mat');
    output_file = fullfile(main_folder, 'step2b_data_output_getGoodBlinkMask.mat');

    data = load(input_file);  % Loads blinkComp, blinkPositions
    zThresholds= data.zThresholds; 
    specifiedStd = data.specifiedStd;
    specifiedMedian= data.specifiedMedian;
    blinkFits= data.blinkFits;
    [goodBlinkMask, specifiedMedian, specifiedStd] = ...
      get_good_blink_mask(blinkFits, specifiedMedian, specifiedStd, zThresholds);

    data_output = load(output_file);
    
    goodBlinkMask_output=data_output.goodBlinkMask;
    specifiedMedian_output=data_output.specifiedMedian;
    specifiedStd_output=data_output.specifiedStd;

    % [areStructsEqual, diffDetails] = compareblinkpropertiesstructure(struct1, struct2)
    findingx=isequal(goodBlinkMask_output,goodBlinkMask) % Return True 1 if same
    g=1
end