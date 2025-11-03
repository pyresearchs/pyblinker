function process_apply_pavr_restriction_step2d()
% PROCESS_APPLY_PAVR_RESTRICTION_STEP2D
% -------------------------------------------------------------------------
% Reference / gold-standard runner for **STEP 2d – applyPAVRRestriction**
% in the Blinker pipeline.
%
% Purpose:
%   This step applies the PAVR-based restriction to the already-computed
%   blink properties and blink fits. In the original MATLAB Blinker code,
%   this comes right after we have computed blink-level features
%   (see STEP 2c: computeBlinkProperties). The idea is to filter out blinks
%   that do not satisfy certain amplitude/velocity/ratio constraints.
%
% Why we keep this as a separate script:
%   - In MATLAB the restriction is often "just another call" at the end of
%     a bigger script.
%   - For the MATLAB → Python migration (pyblinker), we want each step
%     to be reproducible and testable in isolation.
%   - This script therefore becomes the **canonical, testable** version of
%     "run PAVR restriction on these blinkProps + blinkFits + params".
%
% What this script does:
%   1. Resolve paths and load `config.m` (so we can use project-specific
%      folders like `main_folder`)
%   2. Load the step-2d **input fixture**:
%        step2d_data_input_applyPAVRRestriction.mat
%      which should contain:
%        - signalData
%        - params
%        - blinkProps
%        - blinkFits
%   3. Call the actual function under test:
%        [blinkProps, blinkFits] = applyPAVRRestriction(...)
%   4. (Optional) save or display results for later comparison
%
% Recommended filename:
%   process_apply_pavr_restriction_step2d.m
% -------------------------------------------------------------------------

    % ---------------------------------------------------------------------
    % 1. Resolve paths relative to THIS file
    % ---------------------------------------------------------------------
    this_file = mfilename('fullpath');
    this_dir  = fileparts(this_file);
    project_root = fileparts(fileparts(this_dir));  % go up two levels

    data_dir_default   = fullfile(project_root, 'migration_files');
    output_dir_default = fullfile(project_root, 'migration_files');
    if ~exist(output_dir_default, 'dir')
        mkdir(output_dir_default);
    end

    % ---------------------------------------------------------------------
    % 2. Optionally load config.m (overrides defaults if present)
    % ---------------------------------------------------------------------
    config_file = fullfile(this_dir, 'config.m');
    if exist(config_file, 'file')
        run(config_file);
    end

    % Decide actual data directory
    if exist('main_folder', 'var') && isfolder(main_folder)
        data_dir = main_folder;
    else
        data_dir = data_dir_default;
    end

    % ---------------------------------------------------------------------
    % 3. Build input file path (fixture produced by previous steps)
    % ---------------------------------------------------------------------
    input_file = fullfile(data_dir, 'step2d_data_input_applyPAVRRestriction.mat');
    assert(isfile(input_file), 'Input .mat not found: %s', input_file);

    % ---------------------------------------------------------------------
    % 4. Load input data (what we will pass to applyPAVRRestriction)
    % ---------------------------------------------------------------------
    in_data    = load(input_file);
    signalData = in_data.signalData;
    params     = in_data.params;
    blinkProps = in_data.blinkProps;
    blinkFits  = in_data.blinkFits;
    g=1
    % ---------------------------------------------------------------------
    % 5. Run the actual function under test
    % ---------------------------------------------------------------------
    [blinkProps_restricted, blinkFits_restricted] = ...
        applyPAVRRestriction(blinkProps, blinkFits, params, signalData);

    % ---------------------------------------------------------------------
    % 6. (Optional) Save the computed result for comparison / debugging
    % ---------------------------------------------------------------------
    % computed_out_file = fullfile(data_dir, ...
    %     'process_apply_pavr_restriction_step2d_computed_output.mat');
    % save(computed_out_file, ...
    %     'blinkProps_restricted', 'blinkFits_restricted', ...
    %     'blinkProps', 'blinkFits', 'params', 'signalData');
    % 
    % fprintf('PAVR restriction applied and results saved to:\n  %s\n', ...
    %     computed_out_file);
end
