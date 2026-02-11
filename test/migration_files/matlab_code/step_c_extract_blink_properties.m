



function step_c_get_()
    % step_a_get_blink_position
    % Reads configuration from setting.yaml, loads input data,
    % runs extractBlinks (via extractBlinksVersionCompact), and saves blinkPositions.

    % Determine the directory of this script
    currentFile = mfilename('fullpath');
    currentDir  = fileparts(currentFile);

    % Add the current directory to path to ensure we can call sibling functions
    addpath(currentDir);

    yamlFile = fullfile(currentDir, 'setting.yaml');

    % ---- Simple YAML read (supports: key: "value" or key: value) ----
    cfg = read_simple_yaml_kv(yamlFile, {'input_file_step_a_extract_blinks_eeg','output_file_step_c_blink_properties'});
    inputFile  = cfg.input_file_step_a_extract_blinks_eeg;
    outputFile = cfg.output_file_step_c_blink_properties;

    fprintf('Input:  %s\n', inputFile);
    fprintf('Output: %s\n', outputFile);


    data = load(inputFile);

    % Ensure that the required variables are present
    if ~isfield(data, 'blinkComp') || ~isfield(data, 'params')
        error('The input file must contain blinkComp and params variables.');
    end

    % Prepare arguments for extractBlinks
    candidateSignals = data.blinkComp;

    % Ensure candidateSignals is row vector (1 x N)
    if size(candidateSignals, 1) > size(candidateSignals, 2)
        candidateSignals = candidateSignals.';
    end

    % Ensure single precision
    candidateSignals = single(candidateSignals);

    params = data.params;
    signalType = 'SignalNumbers';

    % Call the extraction function
    fprintf('Running extractBlinksVersionCompact...\n');
    [blinks, params] = extractBlinks(candidateSignals, signalType, params);

    signalData=blinks.signalData;

    blink_posi=signalData.blinkPositions;
    blink_posi = signalData.blinkPositions;

    %% 
    assert(ismatrix(blink_posi) && isequal(size(blink_posi), [2 495]), ...
        'Expected blink_posi to be 2x495, got %dx%d.', size(blink_posi,1), size(blink_posi,2));


    
    [blinkProperties, blinkFits] = extractBlinkProperties(signalData, params);
    % Save the output
    outputDir = fileparts(outputFile);
    if ~isempty(outputDir) && ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    save(outputFile, 'blinkProperties', 'blinkFits', '-v7');
    fprintf('Successfully saved blinkProperties and blinkFits to %s\n', outputFile);
end

