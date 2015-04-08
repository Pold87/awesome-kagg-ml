% data = csvread('train.csv', 100, 100, [1 1 3 3]);
%% 

filename = 'test.csv'
fid = fopen(filename);
data = fread(fid, '*char')';  % read all contents into data as a char array (don't forget the `'` to make it a row rather than a column).
fclose(fid);
entries = regexp(data, '\n', 'split');

parsed = regexp(entries(1), ',', 'split');
header = header{1};
id_ = header(1);

n = length(entries) - 1; % last line is wrongly parsed
if filename == 'train.csv'
   size = n - 1;
else
   size = n
end

if size == n - 1
   target = parsed(end);
   header = parsed(2:end - 1);
else
   header = header(2:end);
end

id = zeros(n - 1, 1);
features = zeros(n - 1, length(header));
classes = cell(n - 1, 1);

for i = 2:n
   if mod(i, 100) == 0
      disp(i)
   end
   parsed = regexp(entries(i), ',', 'split');
   parsed = parsed{1};
   id(i - 1) = cellfun(@str2num, parsed(1));
   features(i - 1, :) = cellfun(@str2num, parsed(2:size));
   if size == n - 1
      classes(i - 1) = parsed(end);
   end
end





