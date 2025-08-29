function result = compare_matrices(A, B)
    % Initialize an empty structure array for the result
    result = struct('Status', [], 'Details', []);
    
    % Check if input matrices are the same size
    if ~isequal(size(A), size(B))
        % If sizes do not match, populate result with an error message and sizes
        result.Status = 'Size Mismatch';
        result.Details = struct('A_Size', size(A), 'B_Size', size(B));
        
        % Print error message
        fprintf('Error: Matrices are of different sizes.\n');
        fprintf('Size of A: %dx%d\n', size(A, 1), size(A, 2));
        fprintf('Size of B: %dx%d\n', size(B, 1), size(B, 2));
        return;
    end
    
    % If sizes match, proceed with element-wise comparison
    result.Status = 'Comparison Results';
    result.Details = [];
    mismatch_count = 0;
    
    % Loop through each element and compare
    for i = 1:size(A, 1)
        for j = 1:size(A, 2)
            if A(i, j) ~= B(i, j)
                mismatch_count = mismatch_count + 1;
                
                % Store details of the mismatch in the result structure
                mismatch_info = struct('Row', i, ...
                                       'Column', j, ...
                                       'A_Value', A(i, j), ...
                                       'B_Value', B(i, j), ...
                                       'Difference', A(i, j) - B(i, j));
                result.Details = [result.Details; mismatch_info];
                
                % Print information on the mismatch
                fprintf('Mismatch at row %d, column %d: A = %.4f, B = %.4f, Difference = %.4f\n', ...
                        i, j, A(i, j), B(i, j), A(i, j) - B(i, j));
            end
        end
    end
    
    % Summary of results
    if mismatch_count == 0
        fprintf('All elements are the same in matrices A and B.\n');
    else
        fprintf('Total mismatches found: %d\n', mismatch_count);
    end
end
