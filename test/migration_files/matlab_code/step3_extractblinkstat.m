function process_extract_blink_statistics_step3()
% PROCESS_EXTRACT_BLINK_STATISTICS_STEP3
% -------------------------------------------------------------------------
% Reference / gold-standard runner for **STEP 3 – extractBlinkStatistics**
% in the Blinker pipeline.
%
% Purpose:
%   After we have:
%     - detected and fitted blinks (STEP 1x),
%     - computed blink properties (STEP 2x),
%     - and possibly applied restrictions (e.g. PAVR, STEP 2d),
%   we finally want to **summarize** everything into a statistics
%   structure/table. That is what `extractBlinkStatistics(...)` does.
%
% This script loads the prepared MATLAB fixture for STEP 3, calls the
% actual `extractBlinkStatistics(...)` function, and exports the result to
% a table (Excel) so we can easily inspect the output or compare it with
% the Python port (`pyblinker`).
%
% What this script does:
%   1. Resolve paths relative to this file
%   2. Optionally load `config.m` so project-specific folders are used
%   3. Initialize EEGLAB silently (if available)
%   4. Load the STEP 3 input fixture:
%        step3_data_input_extractBlinkStatistic.mat
%      which should contain:
%        - blinks
%        - blinkFits
%        - blinkProperties
%        - params
%   5. Call:
%        blinkStatistics = extractBlinkStatistics(...)
%   6. Convert to a table and write to `blinkStatistics.xlsx` (for human
%      inspection / CI artifact)
%
% Recommended filename:
%   process_extract_blink_statistics_step3.m
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

    % decide actual data directory
    if exist('main_folder', 'var') && isfolder(main_folder)
        data_dir = main_folder;
    else
        data_dir = data_dir_default;
    end

    % ---------------------------------------------------------------------
    % 3. Initialize EEGLAB silently (if path known)
    % ---------------------------------------------------------------------
    if exist('eeglab_path', 'var') && isfolder(eeglab_path)
        addpath(genpath(eeglab_path));
        eeglab nogui;
    else
        try
            eeglab nogui;
        catch
            error(['EEGLAB could not be started. ' ...
                'Add EEGLAB to your MATLAB path or define eeglab_path in config.m']);
        end
    end

    % ---------------------------------------------------------------------
    % 4. Build input file path and load data
    % ---------------------------------------------------------------------
    input_file = fullfile(data_dir, 'step3_data_input_extractBlinkStatistic.mat');
    assert(isfile(input_file), 'Input .mat not found: %s', input_file);

    in_data          = load(input_file);
    blinks           = in_data.blinks;
    blinkFits        = in_data.blinkFits;
    blinkProperties  = in_data.blinkProperties;
    params           = in_data.params;

    % ---------------------------------------------------------------------
    % 5. Run the actual function under test
    % ---------------------------------------------------------------------
    blinkStatistics = extractBlinkStatistics( ...
        blinks, blinkFits, blinkProperties, params)

    % ---------------------------------------------------------------------
    % 6. Export to table / Excel for inspection
    % ---------------------------------------------------------------------
    blinkTable = struct2table(blinkStatistics, 'AsArray', true);

    % write into the same (or closest) output dir, not random CWD
    xlsx_file = fullfile(data_dir, 'blinkStatistics.xlsx');
    writetable(blinkTable, xlsx_file);

    fprintf('Blink statistics extracted and written to:\n  %s\n', xlsx_file);
end
