function [blinkProps, blinkFits] = applyPAVRRestriction(blinkProps, blinkFits, params, signalData)
    pAVRs = cellfun(@double, {blinkProps.posAmpVelRatioZero});
    frameMax = cell2mat({blinkFits.maxValue});
    pMask = pAVRs < params.pAVRThreshold & ...
        frameMax < signalData.bestMedian - signalData.bestRobustStd;
    blinkProps(pMask) = [];
    blinkFits(pMask) = [];
end
