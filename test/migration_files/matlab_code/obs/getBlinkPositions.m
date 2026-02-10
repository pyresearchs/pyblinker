function blinkPositions = getBlinkPositions(blinkComponent, srate, stdThreshold)
    % GETBLINKPOSITIONS  Detect blink start and end frames (MATLAB port).
    %
    % Usage:
    %   blinkPositions = getBlinkPositions(blinkComponent, srate, stdThreshold)
    %
    % blinkPositions is 2 x N (Row 1: start, Row 2: end) 1-based indices.

    % Constants
    scalingFactor = 1.4826;
    minEventLen = 0.05; % seconds
    minEventSep = 0; % seconds (temporarily 0 for debugging)

    % Ensure row vector
    if size(blinkComponent, 1) > size(blinkComponent, 2)
        blinkComponent = blinkComponent';
    end

    % 1. Compute detection threshold
    mu = mean(blinkComponent);
    madVal = median(abs(blinkComponent - median(blinkComponent)));
    robustStd = scalingFactor * madVal;
    threshold = mu + stdThreshold * robustStd;
    minBlinkFrames = minEventLen * srate;

    % 2. Find blink candidates
    above = (blinkComponent > threshold);
    if ~any(above)
        blinkPositions = [];
        return;
    end

    % Transitions
    diffAbove = diff([0, above, 0]);
    starts = find(diffAbove == 1);
    ends = find(diffAbove == -1); % No -1 here to match Python's 'ends' logic (exclusive)

    % Validate pairs
    numPairs = min(length(starts), length(ends));
    starts = starts(1:numPairs);
    ends = ends(1:numPairs);

    % Filter by duration
    durations = ends - starts; % Exclusive end matching Python
    keepMask = (durations > minBlinkFrames);
    starts = starts(keepMask);
    ends = ends(keepMask);

    if isempty(starts)
        blinkPositions = [];
        return;
    end

    % 3. Remove close blinks
    % If sep < minEventSep, remove BOTH blinks (as in Python code)
    if length(starts) > 1 && minEventSep > 0
        posMask = true(1, length(starts));
        sep = (starts(2:end) - ends(1:end-1)) / srate;
        closeIndices = find(sep < minEventSep);
        
        posMask(closeIndices) = false;
        posMask(closeIndices + 1) = false;
        
        starts = starts(posMask);
        ends = ends(posMask);
    end

    % Format output
    blinkPositions = [starts; ends];
end
