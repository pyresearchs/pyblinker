function [blinks, params] = extractBlinksVersionCompact(candidateSignals, signalType, params)
    % Initialize the blinks structure
    blinks = initializeBlinks(params);

    % Initialize and populate signal data & get the signal position
    % signalData = initializeSignalDataGetSignalPosition(candidateSignals, signalType, params);
    signalData = arrayfun(@(i) createSignalDataGetSignalPosition(candidateSignals, signalType, params, i), ...
                          1:length(params.signalNumbers));

    % Process each signal to extract blink properties
    signalData = arrayfun(@(data) processSignalDataFitBlinks(data, params), signalData);

    % Reduce candidates based on blink amplitude ratios
    signalData = filterByBlinkAmpRatio(signalData, params);

    % Check if any candidate meets the minimum good blink threshold
    signalData = filterByGoodBlinkThreshold(signalData, params);

    % Filter by good blink ratio criteria
    signalData = filterByGoodRatio(signalData, params);

    % Select the best candidate with the maximum number of good blinks
    blinks = selectBestCandidate(signalData, blinks, params);
end

function blinks = initializeBlinks(params)
    % Set up an empty blink structure with initial parameters
    blinks = createBlinksStructure();
    blinks.srate = params.srate;
    blinks.status = ''; 
end



function data = createSignalDataGetSignalPosition(candidateSignals, signalType, params, index)
    % Create a single signal data structure with blink positions
    data = createSignalDataStructure();
    data.signalType = signalType;
    data.signalNumber = params.signalNumbers(index);
    data.signalLabel = params.signalLabels{index};
    data.signal = candidateSignals(index, :);
    data.blinkPositions = getBlinkPositions(data.signal, params.srate, params.stdThreshold);
    data.numberBlinks = size(data.blinkPositions, 2);
end

function data = processSignalDataFitBlinks(data, params)
    % Process individual signal data and calculate blink properties
    blinkFits = fitBlinks(data.signal, data.blinkPositions);
    if isempty(blinkFits)
        return;
    end
    % Calculate blink amplitude ratio and set additional blink properties
    data.blinkAmpRatio = calculateBlinkAmpRatio(data.signal, blinkFits);
    [data.bestMedian, data.bestRobustStd, data.cutoff, data.goodRatio, ...
        data.numberGoodBlinks] = calculateBlinkProperties(blinkFits, params);
end

function ratio = calculateBlinkAmpRatio(signal, blinkFits)
    % Calculate the blink amplitude ratio based on frames inside and outside blinks
    [insideBlink, outsideBlink] = getBlinkRegions(signal, blinkFits);
    ratio = mean(signal(insideBlink)) / mean(signal(outsideBlink));
end

function [bestMedian, bestRobustStd, cutoff, goodRatio, numGoodBlinks] = calculateBlinkProperties(blinkFits, params)
    % Calculate the median, robust standard deviation, cutoff, and good ratio of blinks
    [leftR2, rightR2, maxValues] = extractBlinkParameters(blinkFits);
    goodMaskTop = leftR2 >= params.correlationThresholdTop & rightR2 >= params.correlationThresholdTop;
    goodMaskBottom = leftR2 >= params.correlationThresholdBottom & rightR2 >= params.correlationThresholdBottom;
    if sum(goodMaskTop) < 2
        return;
    end
    [bestMedian, bestRobustStd, cutoff, goodRatio] = calculateGoodBlinkRatios(maxValues, goodMaskTop, goodMaskBottom, params);
    numGoodBlinks = sum(goodMaskBottom);
end

function signalData = filterByBlinkAmpRatio(signalData, params)
    % Filter candidates based on the blink amplitude ratio
    blinkAmpRatios = [signalData.blinkAmpRatio];
    goodIndices = blinkAmpRatios >= params.blinkAmpRange(1) & ...
                  blinkAmpRatios <= params.blinkAmpRange(2);
    if sum(goodIndices) == 0 || isempty(goodIndices)
        blinks.status = 'failure: Blink amplitude too low -- may be noise';
        return;
    end
    signalData = signalData(goodIndices);
end

function signalData = filterByGoodBlinkThreshold(signalData, params)
    % Filter signals by the minimum good blink threshold
    goodCandidates = [signalData.numberGoodBlinks] > params.minGoodBlinks;
    if sum(goodCandidates) == 0
        blinks.status = ['failure: fewer than ' num2str(params.minGoodBlinks) ' were found'];
        return;
    end
    signalData = signalData(goodCandidates);
end

function signalData = filterByGoodRatio(signalData, params)
    % Filter signals by the good blink ratio criteria
    goodRatios = [signalData.goodRatio];
    ratioIndices = goodRatios >= params.goodRatioThreshold;
    if sum(ratioIndices) == 0
        blinks.status = 'failure: Good ratio too low';
        return;
    elseif ~params.keepSignals
        signalData = signalData(ratioIndices);
    end
end

function blinks = selectBestCandidate(signalData, blinks, params)
    % Select the candidate with the maximum number of good blinks
    goodBlinks = [signalData.numberGoodBlinks];
    [~, maxIndex] = max(goodBlinks);
    if ~isempty(maxIndex)
        blinks.status = 'success';
        blinks.usedSignal = signalData(maxIndex).signalNumber;
        blinks.signalData = signalData;
    end
end

function s = createSignalDataStructure()
    % Create a structure for a signalData structure
    s = struct( ...
        'signalType', NaN, ...
        'signalNumber', NaN, ...
        'signalLabel', NaN, ...
        'numberBlinks', NaN, ...
        'numberGoodBlinks', NaN, ...
        'blinkAmpRatio', NaN, ...
        'cutoff', NaN, ...
        'bestMedian', NaN, ...
        'bestRobustStd', NaN, ...
        'goodRatio', NaN, ...
        'signal', NaN, ...
        'blinkPositions', NaN);
end

function s = createBlinksStructure()
    % Return an empty blink structure
    s = struct('fileName', NaN, ...
               'srate', NaN, ...
               'subjectID', NaN, ...
               'experiment', NaN, ...
               'uniqueName', NaN, ...
               'task', NaN, ...
               'startTime', NaN, ...
               'signalData', NaN, ...
               'usedSignal', NaN, ...
               'status', NaN);
end

function [insideBlink, outsideBlink] = getBlinkRegions(signal, blinkFits)
    % Identify regions inside and outside blinks
    blinkMask = false(1, length(signal));
    leftZero = cellfun(@double, {blinkFits.leftZero});
    rightZero = cellfun(@double, {blinkFits.rightZero});
    for j = 1:length(leftZero)
        if rightZero(j) > leftZero(j)
            blinkMask(leftZero(j):rightZero(j)) = true;
        end
    end
    insideBlink = signal > 0 & blinkMask;
    outsideBlink = signal > 0 & ~blinkMask;
end

function [leftR2, rightR2, maxValues] = extractBlinkParameters(blinkFits)
    % Extract parameters from blink fits
    leftR2 = cellfun(@double, {blinkFits.leftR2});
    rightR2 = cellfun(@double, {blinkFits.rightR2});
    maxValues = cellfun(@double, {blinkFits.maxValue});
end

function [bestMedian, bestRobustStd, cutoff, goodRatio] = calculateGoodBlinkRatios(maxValues, goodMaskTop, goodMaskBottom, params)
    % Calculate the best median, robust standard deviation, cutoff, and good ratio
    bestValues = maxValues(goodMaskTop);
    worstValues = maxValues(~goodMaskBottom);
    goodValues = maxValues(goodMaskBottom);
    
    bestMedian = nanmedian(bestValues);
    bestRobustStd = 1.4826 * mad(bestValues, 1);
    worstMedian = nanmedian(worstValues);
    worstRobustStd = 1.4826 * mad(worstValues, 1);
    
    cutoff = (bestMedian * worstRobustStd + worstMedian * bestRobustStd) / ...
             (bestRobustStd + worstRobustStd);
    allValues = sum(maxValues <= bestMedian + 2 * bestRobustStd & ...
                    maxValues >= bestMedian - 2 * bestRobustStd);
    goodRatio = sum(goodValues <= bestMedian + 2 * bestRobustStd & ...
                    goodValues >= bestMedian - 2 * bestRobustStd) / allValues;
end
