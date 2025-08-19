function blinks = processBlinkSignalsCompact(signalData, params)
    % Initialize the output structure
    blinks = struct('usedSignal', NaN, 'status', '');

    %% Step 1: Filter based on blink amplitude ratios
    blinkAmpRatios = cellfun(@double, {signalData.blinkAmpRatio});
    goodIndices = blinkAmpRatios >= params.blinkAmpRange(1) & ...
                  blinkAmpRatios <= params.blinkAmpRange(2);
    if sum(goodIndices) == 0 || isempty(goodIndices)
        blinks.usedSignal = NaN;
        blinks.status = 'failure: Blink amplitude too low -- may be noise';
        return;
    end
    signalData = signalData(goodIndices);

    %% Step 2: Filter based on minimum good blinks
    candidates = cellfun(@double, {signalData.numberGoodBlinks});
    goodCandidates = candidates > params.minGoodBlinks;
    if sum(goodCandidates) == 0
        blinks.status = ['failure: Fewer than ' num2str(params.minGoodBlinks) ' good blinks were found'];
        blinks.usedSignal = NaN;
        return;
    end
    signalData = signalData(goodCandidates);

    %% Step 3: Filter based on good blink ratio criteria
    goodRatios = cellfun(@double, {signalData.goodRatio});
    ratioIndices = goodRatios >= params.goodRatioThreshold;
    testData = signalData; % Default to all signalData
    usedSign = 1; % Flag for success
    if sum(ratioIndices) == 0
        usedSign = -1; % Mark as failure
        blinks.status = 'failure: Good ratio too low';
    elseif ~params.keepSignals
        testData = testData(ratioIndices); % Keep only good ratio candidates
    end

    %% Step 4: Select the candidate with maximum good blinks
    goodBlinks = cellfun(@double, {testData.numberGoodBlinks});
    [~, maxIndex] = max(goodBlinks);
    if usedSign == 1
        blinks.status = 'success: Signal selected';
    end
    blinks.usedSignal = usedSign * testData(maxIndex).signalNumber;
    blinks.signalData = testData;
end