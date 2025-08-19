function [blinkProps, peaksPosVelZero, peaksPosVelBase] = computeBlinkProperties(blinkFits, signalData, params, srate, blinkVelocity, peaks)
    numberBlinks = length(blinkFits);
    
    %% Initialize the blinkProps structure
    blinkProps(numberBlinks) = createPropertiesStructure();
    for k = 1:numberBlinks
        blinkProps(k) = createPropertiesStructure(); %#ok<*AGROW>
    end
    
    peaksPosVelZero = ones(size(peaks));
    peaksPosVelZero(end) = length(signalData.signal);
    peaksPosVelBase = ones(size(peaks));
    peaksPosVelBase(end) = length(signalData.signal);
    signal = signalData.signal;
    
    for k = 1:numberBlinks
        try
            %% Blink durations
            blinkProps(k).durationBase = (blinkFits(k).rightBase - ...
                blinkFits(k).leftBase) ./ srate;
            blinkProps(k).durationTent = (blinkFits(k).rightXIntercept - ...
                blinkFits(k).leftXIntercept) ./ srate;
            blinkProps(k).durationZero = (blinkFits(k).rightZero - ...
                blinkFits(k).leftZero) ./ srate;
            blinkProps(k).durationHalfBase = ...
                (blinkFits(k).rightBaseHalfHeight - ...
                blinkFits(k).leftBaseHalfHeight + 1) ./ srate;
            blinkProps(k).durationHalfZero = ...
                (blinkFits(k).rightZeroHalfHeight - ...
                blinkFits(k).leftZeroHalfHeight + 1) ./ srate;
    
            %% Blink amplitude-velocity ratio from zero to max
            upStroke = blinkFits(k).leftZero:blinkFits(k).maxFrame;
            [~, velFrame] = max(blinkVelocity(upStroke));
            velFrame = velFrame(1) + upStroke(1) - 1;
            peaksPosVelZero(k) = velFrame;
            posAmpVelRatioZero = 100 * abs(signal(blinkFits(k).maxFrame) ...
                ./ blinkVelocity(velFrame)) / srate
            blinkProps(k).posAmpVelRatioZero = 100 * abs(signal(blinkFits(k).maxFrame) ...
                ./ blinkVelocity(velFrame)) / srate;
    
            downStroke = blinkFits(k).maxFrame:blinkFits(k).rightZero;
            aa=blinkVelocity(downStroke)
            [~, velFrame] = min(blinkVelocity(downStroke));
            aaa=downStroke(1)
            velFrame = velFrame(1) + aaa - 1;
            cc=blinkVelocity(velFrame)
            xcx=blinkFits(k).maxFrame
            bbb=signal(xcx)
            qq=bbb./ cc
            negAmpVelRatioZero = 100 * abs(qq) / srate
            blinkProps(k).negAmpVelRatioZero = 100 * abs(signal(blinkFits(k).maxFrame) ...
                ./ blinkVelocity(velFrame)) / srate;
    
            %% Blink amplitude-velocity ratio from base to max
            upStroke = blinkFits(k).leftBase:blinkFits(k).maxFrame;
            [~, velFrame] = max(blinkVelocity(upStroke));
            velFrame = velFrame(1) + upStroke(1) - 1;
            peaksPosVelBase(k) = velFrame;
  
            blinkProps(k).posAmpVelRatioBase = 100 * abs(signal(blinkFits(k).maxFrame) ...
                ./ blinkVelocity(velFrame)) / srate;
    
            downStroke = blinkFits(k).maxFrame:blinkFits(k).rightBase;
            [~, velFrame] = min(blinkVelocity(downStroke));
            velFrame = velFrame(1) + downStroke(1) - 1;
            
            blinkProps(k).negAmpVelRatioBase = 100 * abs(signal(blinkFits(k).maxFrame) ...
                ./ blinkVelocity(velFrame)) / srate;
    
            %% Blink amplitude-velocity ratio estimated from tent slope
            aa=signal(blinkFits(k).maxFrame)
            aaa=blinkFits(k).averRightVelocity
            cc=aa ...
                ./ aaa
            blinkProps(k).negAmpVelRatioTent = 100 * abs(cc) / srate;
            blinkProps(k).posAmpVelRatioTent = 100 * abs(signal(blinkFits(k).maxFrame) ...
                ./ blinkFits(k).averLeftVelocity) / srate;
    
            %% Time zero shut
            blinkProps(k).closingTimeZero = (blinkFits(k).maxFrame - ...
                blinkFits(k).leftZero) ./ srate;
            blinkProps(k).reopeningTimeZero = (blinkFits(k).rightZero - ...
                blinkFits(k).maxFrame) ./ srate;
            thisBlinkAmp = signal(blinkFits(k).leftZero:blinkFits(k).rightZero);
            ampThreshhold = params.shutAmpFraction * blinkFits(k).maxValue;
            startShut = find(thisBlinkAmp >= ampThreshhold, 1, 'first');
            aa_endshut=thisBlinkAmp(startShut+1:end)
            aqa=aa_endshut < ampThreshhold
            endShut = find(aqa, 1, 'first');
            if isempty(endShut)
                blinkProps(k).timeShutZero = 0;
            else
                blinkProps(k).timeShutZero = endShut ./ srate;
            end
    
            %% Time base shut
            thisBlinkAmp = signal(blinkFits(k).leftBase:blinkFits(k).rightBase);
            ampThreshhold = params.shutAmpFraction * blinkFits(k).maxValue;
            startShut = find(thisBlinkAmp >= ampThreshhold, 1, 'first');
            aa=thisBlinkAmp(startShut+1:end)
            aaa=aa < ampThreshhold
            endShut = find(aaa, 1, 'first');
            if isempty(endShut)
                blinkProps(k).timeShutBase = 0;
            else
                blinkProps(k).timeShutBase = endShut ./ srate;
            end
    
            %% Time shut tent
            blinkProps(k).closingTimeTent = (blinkFits(k).xIntersect - ...
                blinkFits(k).leftXIntercept) ./ srate;
            blinkProps(k).reopeningTimeTent = (blinkFits(k).rightXIntercept - ...
                blinkFits(k).xIntersect) ./ srate;

            %%% this agai
            thisBlinkAmp = signal(round(blinkFits(k).leftXIntercept): ...
                round(blinkFits(k).rightXIntercept));
            ampThreshhold = params.shutAmpFraction * blinkFits(k).maxValue;
            startShut = find(thisBlinkAmp >= ampThreshhold, 1, 'first');
            endShut = find(thisBlinkAmp(startShut+1:end) < ampThreshhold, 1, 'first');
            if isempty(endShut)
                blinkProps(k).timeShutTent = 0;
            else
                blinkProps(k).timeShutTent = endShut ./ srate;
            end
    
            %% Other times
            blinkProps(k).peakMaxBlink = blinkFits(k).maxValue;
            blinkProps(k).peakMaxTent = blinkFits(k).yIntersect;
            blinkProps(k).peakTimeTent = blinkFits(k).xIntersect ./ srate;
            blinkProps(k).peakTimeBlink = blinkFits(k).maxFrame ./ srate;



            %%%%%
            blinkProps(k).interBlinkMaxAmp = (peaks(k+1) - peaks(k)) ./ srate;
            blinkProps(k).interBlinkMaxVelBase = (peaksPosVelBase(k+1) - ...
                peaksPosVelBase(k)) ./ srate;
            blinkProps(k).interBlinkMaxVelZero = (peaksPosVelZero(k+1) - ...
                peaksPosVelZero(k)) ./ srate;
        catch Mex
            fprintf('Failed on blink %d: %s\n', k, Mex.message);
        end
    end
    blinkProps(end).interBlinkMaxAmp = NaN;
    blinkProps(end).interBlinkMaxVelBase = NaN;
    blinkProps(end).interBlinkMaxVelZero = NaN;
end

function s = createPropertiesStructure()
    s = struct(...
        'durationBase', nan, ...
        'durationZero', nan, ...
        'durationTent', nan,  ...
        'durationHalfBase', nan, ...
        'durationHalfZero', nan,...
        'interBlinkMaxAmp', nan, ...
        'interBlinkMaxVelBase', nan, ...
        'interBlinkMaxVelZero', nan, ...
        'negAmpVelRatioBase', nan, ...
        'posAmpVelRatioBase', nan, ...
        'negAmpVelRatioZero', nan, ...
        'posAmpVelRatioZero', nan, ...
        'negAmpVelRatioTent', nan, ...
        'posAmpVelRatioTent', nan, ...
        'timeShutBase', nan, ...
        'timeShutZero', nan, ...
        'timeShutTent', nan, ...
        'closingTimeZero', nan, ...
        'reopeningTimeZero', nan, ...
        'closingTimeTent', nan, ...
        'reopeningTimeTent', nan, ...
        'peakTimeBlink', nan,  ...
        'peakTimeTent', nan,  ...
        'peakMaxBlink', nan, ...
        'peakMaxTent', nan);
end