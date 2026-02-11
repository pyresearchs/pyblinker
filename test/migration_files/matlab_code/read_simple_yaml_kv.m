function cfg = read_simple_yaml_kv(yamlFile, keys)
    % READ_SIMPLE_YAML_KV  Simple key-value YAML reader for a flat file.
    % Supports lines like: key: value  OR  key: "value"
    
    cfg = struct();
    for i = 1:length(keys)
        cfg.(keys{i}) = '';
    end
    
    fid = fopen(yamlFile, 'r');
    if fid == -1
        return;
    end
    
    while ~feof(fid)
        line = strtrim(fgetl(fid));
        if isempty(line) || startsWith(line, '#')
            continue;
        end
        
        % Split by first colon
        idx = strfind(line, ':');
        if isempty(idx), continue; end
        
        key = strtrim(line(1:idx(1)-1));
        val = strtrim(line(idx(1)+1:end));
        
        % Remove quotes
        val = regexprep(val, '^[''"]+|[''"]+$', '');
        
        if ismember(key, keys)
            cfg.(key) = val;
        end
    end
    fclose(fid);
end
