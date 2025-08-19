%% Step 2

function step1bii_process_FitBlinks()
    % processBlinkComp loads blinkComp.mat and processes blink positions
        % Load configuration
    config; % Load paths from config.m

    % Define file paths dynamically
    input_file = fullfile(main_folder, 'step1bii_data_input_process_FitBlinks.mat');
    output_file = fullfile(main_folder, 'step1bii_data_output_process_FitBlinks.mat');

    % Load the data from blinkComp.mat
    data = load(input_file);  % Loads blinkComp, blinkPositions
    candidateSignal = data.candidateSignal; 
    blinkPositions = data.blinkPositions;

    % Call the fitBlinks function to get blink fit data
    blinkFits = fitBlinks(candidateSignal, blinkPositions);
    
    data_output = load(output_file); 
    blinkFits_output=data_output.blinkFits;
    [areStructsEqual, diffDetails] = compareblinkpropertiesstructure(blinkFits , blinkFits_output);
    % Define the blink index to be plotted (e.g., 1 for the first blink)
    blinkIndex = 3;

    % Get relevant indices for the specified blink
    maxFrame = blinkFits(blinkIndex).maxFrame;
    leftOuter = blinkFits(blinkIndex).leftOuter;
    rightOuter = blinkFits(blinkIndex).rightOuter;
    leftZero = blinkFits(blinkIndex).leftZero;
    rightZero = blinkFits(blinkIndex).rightZero;
    leftBase = blinkFits(blinkIndex).leftBase;
    rightBase = blinkFits(blinkIndex).rightBase;

    % Extract time range for the selected blink event
    timeRange = leftOuter-10:rightOuter+10;
    blinkSegment = candidateSignal(timeRange);
    % blinkComp= candidateSignal

    % Create a plot
    figure;
     plot(timeRange, blinkSegment, 'LineWidth', 1.5, 'Color', [0, 0, 0, 0.4]); % RGBA color for transparency
    hold on;

    % Plot and annotate each relevant point
    plot(maxFrame, candidateSignal(maxFrame), 'ro', 'MarkerSize', 8, 'DisplayName', 'Max Frame');
    text(maxFrame, candidateSignal(maxFrame), ' Max Frame', 'VerticalAlignment', 'top');

    plot(leftOuter, candidateSignal(leftOuter), 'go', 'MarkerSize', 8, 'DisplayName', 'Left Outer');
    text(leftOuter, candidateSignal(leftOuter), ' Left Outer', 'VerticalAlignment', 'bottom');

    plot(rightOuter, candidateSignal(rightOuter), 'go', 'MarkerSize', 8, 'DisplayName', 'Right Outer');
    text(rightOuter, candidateSignal(rightOuter), ' Right Outer', 'VerticalAlignment', 'bottom');

    plot(leftZero, candidateSignal(leftZero), 'mo', 'MarkerSize', 8, 'DisplayName', 'Left Zero');
    text(leftZero, candidateSignal(leftZero), ' Left Zero', 'VerticalAlignment', 'top');

    plot(rightZero, candidateSignal(rightZero), 'mo', 'MarkerSize', 8, 'DisplayName', 'Right Zero');
    text(rightZero, candidateSignal(rightZero), ' Right Zero', 'VerticalAlignment', 'top');

    plot(leftBase, candidateSignal(leftBase), 'co', 'MarkerSize', 8, 'DisplayName', 'Left Base');
    text(leftBase, candidateSignal(leftBase), ' Left Base', 'VerticalAlignment', 'bottom');

    plot(rightBase, candidateSignal(rightBase), 'co', 'MarkerSize', 8, 'DisplayName', 'Right Base');
    text(rightBase, candidateSignal(rightBase), ' Right Base', 'VerticalAlignment', 'bottom');
        % Overlay scatter plot on the line plot
    scatter(timeRange, blinkSegment, 15, 'b', 'filled'); % Scatter with blue filled markers

    % Label and legend setup
    xlabel('Frame');
    ylabel('Blink Amplitude');
    title(['Blink Analysis for Blink ', num2str(blinkIndex)]);
    legend('show');
    grid on;
    hold off;
    j=1
end
