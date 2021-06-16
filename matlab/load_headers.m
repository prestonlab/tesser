function headers = load_headers(data_dir)
%LOAD_HEADERS   Load all header files.
%
%  headers = load_headers(data_dir)

d = dir(fullfile(data_dir, 'tesserScan_*'));
if isempty(d)
    error('No data found in %s', data_dir);
end

headers = {};
for i = 1:length(d)
    if ~d(i).isdir
        continue
    end
    
    subj_dir = fullfile(data_dir, d(i).name);
    dh = dir(fullfile(subj_dir, 'tesserScan_*_header.mat'));
    if isempty(dh)
        error('No header found in %s', subj_dir);
    elseif length(dh) > 1
        error('Multiple header file matches found in %s', subj_dir)
    end
    header_file = fullfile(subj_dir, dh(1).name);
    headers{i} = getfield(load(header_file, 'header'), 'header');
end
