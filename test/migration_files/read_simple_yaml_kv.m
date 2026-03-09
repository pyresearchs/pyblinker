function cfg = read_simple_yaml_kv(yamlPath, requiredKeys)
    % Minimal YAML parser for flat "key: value" pairs (no nesting, no lists).
    % - Ignores blank lines and comments starting with '#'
    % - Accepts quoted or unquoted values
    % - Keeps backslashes (Windows paths) as-is

    if nargin < 2, requiredKeys = {}; end

    if ~exist(yamlPath, 'file')
        error('Cannot find YAML file: %s', yamlPath);
    end

    txt = fileread(yamlPath);

    % Remove full-line comments and blank lines
    lines = regexp(txt, '\r?\n', 'split');
    lines = lines(:);
    keep = ~cellfun(@(s) isempty(strtrim(s)) || startsWith(strtrim(s), '#'), lines);
    lines = lines(keep);

    cfg = struct();

    % Parse: key : value   (value may contain ':' and backslashes)
    for i = 1:numel(lines)
        line = strtrim(lines{i});

        % Strip inline comment (simple: anything after ' #' or starting '#')
        % If you need '#' inside values, remove this block.
        hashPos = regexp(line, '\s#', 'once');
        if ~isempty(hashPos)
            line = strtrim(line(1:hashPos-1));
        end

        m = regexp(line, '^(?<key>[A-Za-z0-9_]+)\s*:\s*(?<val>.*)$', 'names', 'once');
        if isempty(m), continue; end

        val = strtrim(m.val);

        % Remove surrounding single or double quotes
        if numel(val) >= 2
            if (val(1) == '"'  && val(end) == '"') || (val(1) == '''' && val(end) == '''')
                val = val(2:end-1);
            end
        end

        cfg.(m.key) = val;
    end

    % Validate required keys
    for k = 1:numel(requiredKeys)
        rk = requiredKeys{k};
        if ~isfield(cfg, rk) || isempty(cfg.(rk))
            error('Missing required key "%s" in %s', rk, yamlPath);
        end
    end
end
