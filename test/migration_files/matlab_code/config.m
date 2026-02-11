%% ======================= CONFIGURATION FILE ==========================
% This file defines paths and settings for the EEG preprocessing scripts.
%
% EXPECTED FOLDER STRUCTURE (relative to the GitHub repo root "pyblinker"):
%
% pyblinker/
% ├── test/
% │   ├── test_files/            % raw EEG files (.edf, .bdf, etc.)
% │   │     └── mne_sample_audvis_raw.edf
% │   └── migration_files/       % generated MATLAB outputs
% │         ├── matlab_code/     % MATLAB scripts (this folder)
% │         │     ├── config.m
% │         │     └── step0_process_EEG.m
% │         └── (auto-created outputs like step0_data_input_allChannels_popblinker.mat)
%
% NOTE:
% - These paths are built automatically; no need to hard-code drive letters.
% - Only change `eeglab_path` if EEGLAB is in a custom location on your machine.
%% =====================================================================

% === Locate project structure automatically ===
here = fileparts(mfilename('fullpath'));        % .../matlab_code
migration_root = fileparts(here);               % .../migration_files
test_root      = fileparts(migration_root);     % .../test

% === Define main folders ===
main_folder  = migration_root;                  % where processed data are saved
blinker_dir  = main_folder;                     % Blinker output folder
raw_data_dir = fullfile(test_root, 'test_files'); % where EDF input files are located

% === EEGLAB path (edit this if needed) ===
% Put your local EEGLAB installation path here.
% If this folder does not exist, the script will skip adding EEGLAB.
eeglab_path = 'D:\code_development\matlab_plugin\eeglab2025.1.0';

% ======================================================================
% You normally do NOT need to edit anything below this line.
% Paths are relative to the repo so it works across computers.
% ======================================================================
