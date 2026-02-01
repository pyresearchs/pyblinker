"""
We try to immitate the behavior of the first stage of the blinker
In the first major part, under the pop_blinker, as shown in step0_pop_blinker.m,
we have call function name
[blinks, params] = extractBlinksEEG(EEG, params);
and it will output parameter such as
numberBlinks,numberGoodBlinks, blinkAmpRatio, cutoff,bestMedian,bestRobustStd,blinkPositions.
To avoid change the MATLAB code,and we also want to validate (all except fitBliks as we already have a dedicated file to validate it)

   +--------------------------------------------------------------+
        |        Step 3A: extractBlinks(...) (Candidate Selection)      |
        |                 [Loop over signalData(k)]                     |
        |                                                              |
        |   ✓ fitBlinks per candidate                                   |
        |   ✓ Compute blinkAmpRatio / goodRatio / numberGoodBlinks      |
        |   ✓ Filter by blinkAmpRange                                   |
        |   ✓ Filter by minGoodBlinks                                   |
        |   ✓ Apply goodRatioThreshold (may set usedSign=-1)            |
        |   ✓ Pick max(numberGoodBlinks) -> final used signal

so to do this, we may need to run to whole process, and compare with the MATLAB .mat output.

However, to avoid the need for downsampling, which mau cause other disprepancy, we will try to run the process without downsampling first. Meaning, we will skip
The following
if nargin < 2
    params = struct();
end

[params, errors] = checkBlinkerDefaults(params, getBlinkerDefaults(EEG));
if ~isempty(errors)
    error('extractBlinks:BadParameters', ['|' sprintf('%s|', errors{:})]);
end

%% Extract the candidate signals
if params.verbose
    fprintf('Extracting candidate signals...\n');
end
[candidateSignals, signalType, signalNumbers, ...
                signalLabels, params] = getCandidateSignals(EEG, params);
params.signalNumbers = signalNumbers;
params.signalLabels = signalLabels;
if params.verbose
    fprintf('Extracting blinks from the candidate signals... be patient....\n');
end

and directly do everything that
[blinks, params] = extractBlinks(candidateSignals, signalType, params);

The

"""