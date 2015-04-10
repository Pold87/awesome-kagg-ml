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

load('train.mat');

trainfeatures = features(1:2:end, :);
trainlabelscell = classes(1:2:end, :);
testfeatures = features(1:2:end, :);
testlabelscell = classes(1:2:end, :);


% only works for numbers of 1 digit
trainlabels = arrayfun(@(class) str2num(class{1}(end)), trainlabelscell);
testlabels  = arrayfun(@(class) str2num(class{1}(end)), testlabelscell);

y_test = repmat(testlabels, 1, 9) == repmat(1:9, length(testlabels), 1);
y_train = repmat(trainlabels, 1, 9) == repmat(1:9, length(trainlabels), 1); 

%% newff

trainset = trainfeatures(1:end, :)';
trainsetlabels = y_train(1:end, :)';

testset = testfeatures(1:end, :)';
testsetlabels = y_test(1:end, :)';

n_layers = 3

net = newff(trainset, trainsetlabels, n_layers);
net = train(net, trainset, trainsetlabels);
p_test = net(testset);
% p_test = max(p_test, 0);
tmp = (p_test - repmat(min(p_test), 9, 1)) ./ (repmat(max(p_test), 9, 1) - repmat(min(p_test), 9, 1));
ll = logloss(tmp', testsetlabels')


%% perceptron

trainset = trainfeatures(1:end, :)';
trainsetlabels = y_train(1:end, :)';

testset = testfeatures(1:end, :)';
testsetlabels = y_test(1:end, :)';

net = perceptron
net = train(net, trainset, trainsetlabels)

p_test = net(testset);
% p_test = max(p_test, 0);
tmp = (p_test - repmat(min(p_test), 9, 1)) ./ (repmat(max(p_test), 9, 1) - repmat(min(p_test), 9, 1));
ll = logloss(tmp', testsetlabels')




%% test logloss

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


