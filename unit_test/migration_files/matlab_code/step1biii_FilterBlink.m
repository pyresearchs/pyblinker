function processFitBlinks()
    % For step FilterBlink (STEP 1biii) # I creatively create this function name as the Blinker 
    % just lay the calculation under the 2.	extractBlinksEEG code (after
    % extractBlinks). Based on this understanding, when executing this
    % understanding code, you just mark a debug mode on line 72, let the
    % code process both the getBlinkPositions (STEP 1bi)  and fitBlinks
    % (STEP 1bii) since I am to lazy to refactor the code after line 71 as
    % a single block code
    % Load configuration
    config; % Load paths from config.m
    input_file = fullfile(main_folder, 'step1biii_data_input_process_step_FilterBlink.mat');
    data = load(input_file);  % Loads blinkComp, blinkPositions
    candidateSignals= data.candidateSignals; 
    params = data.params;
    signalType= data.signalType;
    
    % Although we could use AI to decompose the blink filtration process into 
    % individual steps, such as:
    %   - filterByBlinkAmpRatio (STEP 1biii) 
    %   - filterByGoodBlinkThreshold (STEP 1biv) 
    %   - filterByGoodRatio (STEP 1bv) 
    %   - selectBestCandidate (STEP 1bvi)
    % we want to avoid the risk of errors from potential reordering.
    %
    % Therefore, to ensure accuracy when evaluating the output of blink filtration,
    % we will run the entire extractBlinks logic here. This approach will 
    % also evaluate the GetSignalPosition and FitBlinks functions.
    %
    % In the Python implementation, however, we will decompose the code as 
    % proposed, following the steps: STEP 1biii, STEP 1biv, STEP 1bv, and STEP 1bvi.

    [blinks, params] = extractBlinks(candidateSignals, signalType, params);
    data_output = load('C:\Users\balan\IdeaProjects\pyblinker\Devel\step1biii_data_output_process_step_FilterBlink.mat');
    blinks_output=data_output.blinks;
    [areStructsEqual, diffDetails] = compareblinkpropertiesstructure(blinks.signalData, blinks_output.signalData)
 
end