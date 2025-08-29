function step2d_applyPAVRRestriction()

    % Load configuration
    config; % Load paths from config.m

    % Define file paths dynamically
    input_file = fullfile(main_folder, 'step2d_data_input_applyPAVRRestriction.mat');
    % output_file = fullfile(main_folder, 'step1bi_data_output_getBlinkPositions.mat');


    data = load(input_file);  % Loads blinkComp, blinkPositions

    signalData= data.signalData;
    params= data.params;
    blinkProps= data.blinkProps;
    blinkFits= data.blinkFits;
    [blinkProps, blinkFits] = applyPAVRRestriction(blinkProps, blinkFits, params, signalData);
    h=1
end