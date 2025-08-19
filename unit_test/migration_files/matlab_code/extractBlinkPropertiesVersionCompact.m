function [blinkProps, blinkFits] = extractBlinkPropertiesVersionCompact(signalData, params)
% Return a structure with blink shapes and properties for individual blinks
%
% Parameters:
%     signalData    signalData structure
%     params        params structure with parameters
%     blinkProps    (output) structure with the blink properties
%     blinkFits     (output) structure with the blink landmarks
%
% BLINKER extracts blinks and ocular indices from time series.
% Copyright (C) 2016  Kay A. Robbins, Kelly Kleifgas, UTSA
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

%% Compute the fits
srate = params.srate;
signal = signalData.signal;
% [STEP 2A]
blinkFits = fitBlinks(signal, signalData.blinkPositions);
if isempty(blinkFits)
    blinkProps = '';
    return;
end

%% First reduce on the basis of blink maximum amplitude 
% [STEP 2B]
goodBlinkMask = get_good_blink_mask(blinkFits, signalData.bestMedian, ...
                 signalData.bestRobustStd, params.zThresholds);
blinkFits = blinkFits(goodBlinkMask);
if isempty(blinkFits)
    blinkProps = '';
    return;
end

%% Compute the blink properties
blinkVelocity = diff(signal);
peaks = cell2mat({blinkFits.maxFrame});
peaks = [peaks length(signal)];

% [STEP 2C]
[blinkProps, peaksPosVelZero, peaksPosVelBase] = computeBlinkProperties(blinkFits, signalData, params, srate, blinkVelocity, peaks);

%% Now apply the final restriction on pAVR to reduce the eye movements
% [STEP 2D]
[blinkProps, blinkFits] = applyPAVRRestriction(blinkProps, blinkFits, params, signalData);




end
