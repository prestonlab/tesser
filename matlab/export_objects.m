function tab = export_objects(data_dir, output_file)
%EXPORT_OBJECTS   Write a table of object mappings.
%
% tab = export_objects(data_dir, output_file)

headers = load_headers(data_dir);
subject = [];
node = [];
object = [];
n_node = 21;
for i = 1:length(headers)
    subject = [subject; repmat(headers{i}.subjNum, [n_node, 1])];
    node = [node; [1:n_node]'];
    pics = headers{i}.parameters.picshuf;
    for j = 1:length(pics)
        [~, name, ~] = fileparts(pics{j});
        parts = regexp(name, '_', 'split');
        number = str2num(parts{2});
        object = [object; number];
    end
end

tab = table(subject, node, object);
writetable(tab, output_file);
