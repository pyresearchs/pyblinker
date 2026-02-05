function step_a_get_blink_position()
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
    cfg = read_simple_yaml_kv(yamlFile, {'input_file_step_a_blink_posi','output_file_step_blink_posi'});
    inputFile  = cfg.input_file_step_a_blink_posi;
    outputFile = cfg.output_file_step_blink_posi;

    fprintf('Input:  %s\n', inputFile);
    fprintf('Output: %s\n', outputFile);

    % Load the input data
    if ~exist(inputFile, 'file')
        error('Input file does not exist: %s', inputFile);
    end
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
    [blinks, params] = extractBlinksVersionCompact(candidateSignals, signalType, params);

    % Extract blinkPositions
    % if isfield(blinks, 'signalData') && ~isempty(blinks.signalData) && isfield(blinks.signalData(1), 'blinkPositions')
    %     blinkPositions = blinks.signalData(1).blinkPositions;
    % else
    %     warning('No blinkPositions found in blinks output. Saving empty blinkPositions.');
    %     blinkPositions = [];
    % end

    % Save the output
    outputDir = fileparts(outputFile);
    if ~isempty(outputDir) && ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    save(outputFile, 'blinks', '-v7');
    fprintf('Successfully saved blinkPositions to %s\n', outputFile);
end

