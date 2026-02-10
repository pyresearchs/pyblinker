function blinkFits = fitBlinks(signal, blinkPositions)
    % FITBLINKS  Simplified blink fitting for migration validation.
    % Only extracts maxValue, leftR2, rightR2 to satisfy extractBlinksVersionCompact.
    
    numBlinks = size(blinkPositions, 2);
    blinkFits = struct('maxValue', cell(1, numBlinks), ...
                       'maxFrame', cell(1, numBlinks), ...
                       'leftR2', cell(1, numBlinks), ...
                       'rightR2', cell(1, numBlinks), ...
                       'leftZero', cell(1, numBlinks), ...
                       'rightZero', cell(1, numBlinks));
    
    dataSize = length(signal);
    
    for i = 1:numBlinks
        startIdx = blinkPositions(1, i);
        endIdx = blinkPositions(2, i);
        
        % 1. Get max blink
        [maxVal, maxFrame] = get_max_blink_local(signal, startIdx, endIdx);
        blinkFits(i).maxValue = maxVal;
        blinkFits(i).maxFrame = maxFrame;
        
        % 2. Get zero crossings (for regions)
        outerStart = 1;
        if i > 1
            [~, prevMaxFrame] = get_max_blink_local(signal, blinkPositions(1, i-1), blinkPositions(2, i-1));
            outerStart = prevMaxFrame;
        end
        
        outerEnd = dataSize;
        if i < numBlinks
            [~, nextMaxFrame] = get_max_blink_local(signal, blinkPositions(1, i+1), blinkPositions(2, i+1));
            outerEnd = nextMaxFrame;
        end
        
        [leftZero, rightZero] = left_right_zero_crossing_local(signal, maxFrame, outerStart, outerEnd);
        blinkFits(i).leftZero = leftZero;
        blinkFits(i).rightZero = rightZero;
        
        % 3. Simplified R2 calculation (Linear fit on sides)
        % Left side: from leftZero to maxFrame
        blinkFits(i).leftR2 = compute_linear_r2(signal, leftZero, maxFrame);
        
        % Right side: from maxFrame to rightZero
        blinkFits(i).rightR2 = compute_linear_r2(signal, maxFrame, rightZero);
    end
end

function [maxVal, maxFrame] = get_max_blink_local(signal, startIdx, endIdx)
    [maxVal, idx] = max(signal(startIdx:endIdx));
    maxFrame = startIdx + idx - 1;
end

function [leftZero, rightZero] = left_right_zero_crossing_local(signal, maxFrame, outerStart, outerEnd)
    % Simplified zero crossing
    % Left
    leftZero = maxFrame;
    for j = maxFrame:-1:outerStart
        if signal(j) <= 0
            leftZero = j;
            break;
        end
        if signal(j) < signal(leftZero)
            leftZero = j;
        end
    end
    
    % Right
    rightZero = maxFrame;
    for j = maxFrame:outerEnd
        if signal(j) <= 0
            rightZero = j;
            break;
        end
        if signal(j) < signal(rightZero)
            rightZero = j;
        end
    end
end

function r2 = compute_linear_r2(signal, startIdx, endIdx)
    if startIdx >= endIdx
        r2 = 0;
        return;
    end
    x = (startIdx:endIdx)';
    y = signal(startIdx:endIdx)';
    
    if length(x) < 2
        r2 = 0;
        return;
    end
    
    % Linear fit y = ax + b
    p = polyfit(x, y, 1);
    y_fit = polyval(p, x);
    
    y_mean = mean(y);
    ss_tot = sum((y - y_mean).^2);
    ss_res = sum((y - y_fit).^2);
    
    if ss_tot == 0
        r2 = 1;
    else
        r2 = 1 - (ss_res / ss_tot);
    end
    
    if r2 < 0, r2 = 0; end
end
