% data = csvread('train.csv', 100, 100, [1 1 3 3]);

%% read/parse the data

filename = 'train.csv'
fid = fopen(filename);
data = fread(fid, '*char')';  % read all contents into data as a char array (don't forget the `'` to make it a row rather than a column).
fclose(fid);
entries = regexp(data, '\n', 'split');

parsed = regexp(entries(1), ',', 'split');
parsed = parsed{1};
id_ = parsed(1);

n = length(entries) - 1; % last line is wrongly parsed
% n = 100;
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

id = zeros(size, 1);
features = zeros(size, length(header));
classes = cell(size, 1);

for i = 2:n
   if mod(i, 100) == 0
      disp(i)
   end
   parsed = regexp(entries(i), ',', 'split');
   parsed = parsed{1};
   id(i - 1) = cellfun(@str2num, parsed(1));
   if size == n - 1
      features(i - 1, :) = cellfun(@str2num, parsed(2:end - 1));
      classes(i - 1) = parsed(end);
   else
      features(i - 1, :) = cellfun(@str2num, parsed(2:end));
   end
end


%% classify

train = features(1:2:end, :);
trainlabelscell = classes(1:2:end, :);
test = features(1:2:end, :);
testlabelscell = classes(1:2:end, :);


% -- bah
trainlabels = zeros(length(trainlabelscell), 1);
for i = 1:length(trainlabelscell)
   trainlabels(i) = str2num(trainlabelscell{i}(end:end));
end

testlabels = zeros(length(testlabelscell), 1);
for i = 1:length(testlabelscell)
   testlabels(i) = str2num(testlabelscell{i}(end:end));
end

y = zeros(length(testlabelscell), 9);
for i = 1:length(testlabels)
   y(i, testlabels(i)) = 1;
end
% -- tot hier bah

net = newff(train, trainlabels, 5);
net = train(net, train, trainlabels);
outputs = net(inputs);

%% 
y = [
   [1, 0, 0, 0, 0, 0, 0, 0, 0];
   [1, 0, 0, 0, 0, 0, 0, 0, 0];
   [0, 1, 0, 0, 0, 0, 0, 0, 0];
   [1, 0, 0, 0, 0, 0, 0, 0, 0];
   [0, 0, 0, 0, 1, 0, 0, 0, 0];
   [0, 1, 0, 0, 0, 0, 0, 0, 0];
   [0, 0, 1, 0, 0, 0, 0, 0, 0];
   [0, 0, 0, 1, 0, 0, 0, 0, 0];
   [0, 0, 0, 1, 0, 0, 0, 0, 0];
   [0, 0, 0, 0, 1, 0, 0, 0, 0];
   [0, 0, 0, 0, 1, 0, 0, 0, 0];
   [0, 0, 0, 0, 0, 1, 0, 0, 0];
   [0, 0, 0, 0, 0, 1, 0, 0, 0];
   [0, 0, 0, 0, 0, 1, 0, 0, 0];
   [0, 0, 0, 0, 0, 0, 1, 0, 0];
   [0, 0, 0, 0, 0, 0, 0, 1, 0];
   [0, 0, 0, 0, 0, 0, 0, 0, 1];
   [0, 0, 0, 0, 0, 0, 0, 0, 1];
   [0, 0, 0, 0, 0, 0, 0, 0, 1];
   [0, 0, 0, 0, 0, 0, 0, 1, 0];
   [0, 0, 0, 0, 0, 0, 1, 0, 0];
   [0, 0, 1, 0, 0, 0, 0, 0, 0];
   [0, 1, 0, 0, 0, 0, 0, 0, 0]];

[n m] = size(y);

p = magic(max(n, m));
p = p(1:n, 1:m);
p = p / max(max(p));

logloss(p, y)


