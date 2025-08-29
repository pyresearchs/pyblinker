function step2c_computeBlinkProperties()

% Load configuration
    config; % Load paths from config.m

    % Define file paths dynamically
    input_file = fullfile(main_folder, 'step2c_data_output_computeBlinkProperties.mat');
    output_file = fullfile(main_folder, 'step2c_data_input_computeBlinkProperties.mat');

    data_output = load(input_file);  % Loads blinkComp, blinkPositions
    
    blinkProps_output=data_output.blinkProps;
    
    data = load(output_file);  % Loads blinkComp, blinkPositions
    blinkFits= data.blinkFits;
    signalData= data.signalData;
    params= data.params;
    srate = data.srate;
    blinkVelocity= data.blinkVelocity;
    peaks= data.peaks;


    [blinkProps, peaksPosVelZero, peaksPosVelBase] = computeBlinkProperties(blinkFits, signalData, params, srate, blinkVelocity, peaks);


    peaksPosVelZero_output=data_output.peaksPosVelZero;
    peaksPosVelBase_output=data_output.peaksPosVelBase;

    [areStructsEqual_blinkProps, diffDetails_blinkProps] = compareblinkpropertiesstructure(blinkProps, blinkProps_output)
    result_peaksPosVelZero = compare_matrices(peaksPosVelZero , peaksPosVelZero_output);
    result_peaksPosVelBase= compare_matrices(peaksPosVelBase , peaksPosVelBase_output);
    h=2
    % elementwise_comparison_peaksPosVelZero = (peaksPosVelZero == peaksPosVelZero_output);
    % elementwise_comparison_peaksPosVelBase = (peaksPosVelBase == peaksPosVelBase_output);

    h=1
end
