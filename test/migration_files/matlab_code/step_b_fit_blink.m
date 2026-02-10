% We have to do the step b fit blink, because the output for
% step_a_extract_blinks_eeg did not produce the output for fit blink. In
% this case, we need the blink position that extracted from step_a as an
% input into the step b.
function step_b_fit_blink()
    % step_b_fit_blink
    % Reads configuration from setting.yaml, loads input data (candidateSignal and blinkPositions),
    % runs fitBlinks, and saves blinkFits.

    % Determine the directory of this script
    currentFile = mfilename('fullpath');
    currentDir  = fileparts(currentFile);

    % Add the current directory to path to ensure we can call sibling functions
    addpath(currentDir);

    yamlFile = fullfile(currentDir, 'setting.yaml');

    % ---- Simple YAML read ----
    cfg = read_simple_yaml_kv(yamlFile, {'input_file_step_a_blink_posi', 'output_file_step_blink_posi', 'output_file_fitblink'});
    inputFileSignal  = cfg.input_file_step_a_blink_posi;
    inputFileBlinks  = cfg.output_file_step_blink_posi;
    outputFile       = cfg.output_file_fitblink;

    fprintf('Input Signal: %s\n', inputFileSignal);
    fprintf('Input Blinks: %s\n', inputFileBlinks);
    fprintf('Output:       %s\n', outputFile);

    % 1. Load the signal (candidateSignal) from the original input file
    if ~exist(inputFileSignal, 'file')
        error('Input signal file does not exist: %s', inputFileSignal);
    end
    dataSignal = load(inputFileSignal);

    if ~isfield(dataSignal, 'blinkComp')
        error('The input signal file must contain blinkComp variable.');
    end
    candidateSignal = dataSignal.blinkComp;

    % Ensure candidateSignal is row vector (1 x N) or appropriate for fitBlinks
    if size(candidateSignal, 1) > size(candidateSignal, 2)
        candidateSignal = candidateSignal.';
    end
    % Ensure single precision if needed (matching step_a)
    candidateSignal = single(candidateSignal);


    % 2. Load the blinkPositions from the previous step output
    if ~exist(inputFileBlinks, 'file')
        error('Input blinks file does not exist: %s', inputFileBlinks);
    end
    dataBlinks = load(inputFileBlinks);

    % Extract blinkPositions
    % Expected structure: dataBlinks.blinks.signalData(1).blinkPositions
    % or similar depending on extractBlinks output.
    
    blinkPositions = [];
    if isfield(dataBlinks, 'blinks')
        blinksStruct = dataBlinks.blinks;
        if isfield(blinksStruct, 'signalData') && ~isempty(blinksStruct.signalData)
             if isfield(blinksStruct.signalData(1), 'blinkPositions')
                blinkPositions = blinksStruct.signalData(1).blinkPositions;
             else
                 warning('blinks.signalData(1) does not have blinkPositions field.');
             end
        elseif isfield(blinksStruct, 'blinkPositions')
             blinkPositions = blinksStruct.blinkPositions;
        end
    elseif isfield(dataBlinks, 'blinkPositions')
        blinkPositions = dataBlinks.blinkPositions;
    end

    if isempty(blinkPositions)
        warning('Could not find blinkPositions in %s. Passing empty.', inputFileBlinks);
    end

    % 3. Call fitBlinks
    fprintf('Running fitBlinks...\n');
    blinkFits = fitBlinks(candidateSignal, blinkPositions);

    % 4. Save the output
    outputDir = fileparts(outputFile);
    if ~isempty(outputDir) && ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    save(outputFile, 'blinkFits', '-v7');
    fprintf('Successfully saved blinkFits to %s\n', outputFile);
end