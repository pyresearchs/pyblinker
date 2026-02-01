```plaintext
        +--------------------------------------------------------------+
        |                 MATLAB Blinker (EEGLAB) Pipeline              |
        |      process_eeg_with_pop_blinker.m -> pop_blinker.m          |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        |          Step 1: process_eeg_with_pop_blinker.m               |
        |                      [Entry Script]                           |
        |                                                              |
        |   ✓ Calls pop_blinker.m                                       |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        |                    Step 2: pop_blinker.m                      |
        |                 [Preprocess + Candidate Signals]              |
        |                                                              |
        |   ✓ Preprocess                                               |
        |   ✓ Build candidateSignals (multiple time series)             |
        |   ✓ Calls: extractBlinksEEG(EEG, params)                      |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        |                Step 3: extractBlinksEEG(...)                  |
        |         [blinks, params] = extractBlinksEEG(EEG, params)      |
        |                                                              |
        |   ✓ Internally calls extractBlinks(...) to score candidates   |
        |   ✓ Selects ONE best candidate signal                         |
        |   ✓ Returns to extractBlinksEEG with chosen signalData        |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        |        Step 3A: extractBlinks(...) (Candidate Selection)      |
        |                 [Loop over signalData(k)]                     |
        |                                                              |
        |   ✓ fitBlinks per candidate                                   |
        |   ✓ Compute blinkAmpRatio / goodRatio / numberGoodBlinks      |
        |   ✓ Filter by blinkAmpRange                                   |
        |   ✓ Filter by minGoodBlinks                                   |
        |   ✓ Apply goodRatioThreshold (may set usedSign=-1)            |
        |   ✓ Pick max(numberGoodBlinks) -> final used signal           |
        |                                                              |
        |   Output:                                                     |
        |     blinks.usedSignal, blinks.signalData (= testData)         |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        |      Step 4: Back in extractBlinksEEG(...) (Finalize)         |
        |                     [Finalize blinks struct]                 |
        |                                                              |
        |   ✓ Uses selected candidate signalData (the chosen one)       |
        |   ✓ Calls extractBlinkProperties(signalData, params)          |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        |            Step 5: extractBlinkProperties.m                   |
        |   [blinkProperties, blinkFits] = extractBlinkProperties(...)  |
        |                                                              |
        |   Input:                                                     |
        |     - signalData (chosen signal only)                         |
        |     - params (srate, thresholds, etc.)                        |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        |              Step 5A: Compute fits (again, per chosen signal) |
        |   blinkFits = fitBlinks(signalData.signal,                    |
        |                         signalData.blinkPositions)            |
        |                                                              |
        |   ├─ if isempty(blinkFits): blinkProps='' ; return            |
        |   └─ else continue                                            |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        |         Step 5B: Reduce blinks by max amplitude (Z-threshold) |
        |   goodBlinkMask = getGoodBlinkMask(blinkFits,                 |
        |                     signalData.bestMedian,                    |
        |                     signalData.bestRobustStd,                 |
        |                     params.zThresholds)                       |
        |   blinkFits = blinkFits(goodBlinkMask)                        |
        |                                                              |
        |   ├─ if isempty(blinkFits): blinkProps='' ; return            |
        |   └─ numberBlinks = length(blinkFits)                         |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        |          Step 5C: Init blinkProps struct array (size N)       |
        |   blinkProps(numberBlinks) = createPropertiesStructure()      |
        |   for k = 1:numberBlinks: blinkProps(k)=createProperties...   |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        |        Step 5D: FOR each blink k (per-blink properties)       |
        |                     [try/catch per blink]                    |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        |      Step 5D-1: Duration features (in seconds = frames/srate) |
        |   durationBase      = (rightBase - leftBase)/srate            |
        |   durationTent      = (rightXIntercept - leftXIntercept)/srate|
        |   durationZero      = (rightZero - leftZero)/srate            |
        |   durationHalfBase  = (rightBaseHalfHeight-leftBaseHalfHeight+1)/srate |
        |   durationHalfZero  = (rightZeroHalfHeight-leftZeroHalfHeight+1)/srate |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        |   Step 5D-2: Amp-Velocity ratios (pAVR/nAVR) from derivatives |
        |   blinkVelocity = diff(signal)                                |
        |   - Find max positive vel frame on upstroke (Zero->Max)       |
        |   - Find min negative vel frame on downstroke (Max->Zero)     |
        |   posAmpVelRatioZero, negAmpVelRatioZero                      |
        |   posAmpVelRatioBase, negAmpVelRatioBase                      |
        |   posAmpVelRatioTent, negAmpVelRatioTent (using tent slopes)  |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        |          Step 5D-3: Time shut metrics (Zero/Base/Tent)         |
        |   closingTimeZero  = (maxFrame - leftZero)/srate              |
        |   reopeningTimeZero= (rightZero - maxFrame)/srate             |
        |   timeShutZero: fraction above shutAmpFraction*maxValue       |
        |   timeShutBase: same, but leftBase:rightBase                  |
        |   timeShutTent: same, but leftXIntercept:rightXIntercept      |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        |             Step 5D-4: Peak + Inter-blink timings             |
        |   peakMaxBlink = maxValue                                     |
        |   peakMaxTent  = yIntersect                                   |
        |   peakTimeTent = xIntersect/srate                              |
        |   peakTimeBlink= maxFrame/srate                                |
        |   interBlinkMaxAmp / interBlinkMaxVelBase / interBlinkMaxVelZero |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        |              Step 5E: Final restriction to reduce eye movement |
        |   pAVRs = [blinkProps.posAmpVelRatioZero]                      |
        |   frameMax = [blinkFits.maxValue]                              |
        |   pMask = (pAVR < params.pAVRThreshold) AND                    |
        |           (frameMax < bestMedian - bestRobustStd)              |
        |   Remove masked blinks: blinkProps(pMask)=[]; blinkFits(pMask)=[] |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        |     Step 6: (Next) extractBlinkStatistics(...)                |
        |   blinkStatistics = extractBlinkStatistics(blinks, blinkFits, |
        |                          blinkProperties, params)             |
        |   (computes summary ocular indices / statistics)              |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        |          Step 7: Return up to extractBlinksEEG -> pop_blinker  |
        |                                                              |
        |   Output (typical):                                           |
        |     blinks (finalized)                                        |
        |     params (final parameters)                                 |
        |     blinkProperties + blinkFits (internal/attached as needed) |
        +--------------------------------------------------------------+
                               |
                               v
        +------------------------------+
        |            End               |
        +------------------------------+
```
