function [areStructsEqual, diffDetails] = compareblinkpropertiesstructure(struct1, struct2)
% compareStructs - Compare two MATLAB structs field by field and element by element
% Usage:
%   [areStructsEqual, diffDetails] = compareStructs(struct1, struct2)
%
% Inputs:
%   struct1 - First struct to compare
%   struct2 - Second struct to compare
%
% Outputs:
%   areStructsEqual - Logical value, true if structs are identical, false otherwise
%   diffDetails - Cell array of differences found (field and index)

    % Initialize result and details
    areStructsEqual = true;
    diffDetails = {};  % Cell array to store differences

    % Get all field names of the structs
    fields1 = fieldnames(struct1);
    fields2 = fieldnames(struct2);

    % Check if both structs have the same fields
    if length(fields1) ~= length(fields2) || ~all(strcmp(fields1, fields2))
        warning('Structs have different fields.');
        areStructsEqual = false;
        return;
    end

    % Loop through each element in the struct array
    for i = 1:length(struct1)
        % Loop through each field to compare values
        for j = 1:numel(fields1)
            field = fields1{j};

            % Get values for each field in the current struct element
            value1 = struct1(i).(field);
            value2 = struct2(i).(field);

            % Check if both values are NaN (NaN is not equal to NaN in MATLAB)
            if (isnumeric(value1) && isnumeric(value2)) && ...
               all(isnan(value1(:))) && all(isnan(value2(:)))
                continue; % Skip this field as both values are NaN
            end

            % Compare values (using isequal for deep comparison)
            if ~isequal(value1, value2)
                % Log the difference
                diffDetails{end+1} = sprintf('Difference in field "%s" at index %d. %d vs %d', field, i, value1, value2); %#ok<AGROW>
                areStructsEqual = false;
            end
        end
    end

    % Display differences if any
    if areStructsEqual
        disp('Both structs are identical.');
    else
        disp('Structs have differences:');
        disp(diffDetails);
    end
end
