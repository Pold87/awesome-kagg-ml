addpath('../drivers/2');

trips = 200;
sequences{trips} = [];

turns = 50;
freqs = 20;
bins = 10;

% cntr = 1;
for i = 1:trips
	trip = csvread([num2str(i) '.csv'], 1, 0);
	% trip2 = csvread([num2str(2) '.csv'], 1, 0);

% 	trip = trip(1:50,:);
% 	figure;
	trip = rotate(trip);
% 	subplot(2, 6, cntr);
% 	plot(trip(:, 1), trip(:, 2))
	
	data = extractAngles(trip);
	data = smoothData(data, turns, freqs);

% 	figure;
% 	bar(data);
	data = binData(data, bins);

% 	figure;
% 	subplot(2, 6, cntr + 6);
% 	bar(data);
% 	cntr = cntr + 1;
	
	sequences{i} = data;

end

% sym_scores = zeros(trips, trips);
% for i = 1:trips
% 	for j = i + 1:trips
% 		sym_scores(i, j) = seqalign(sequences{i}, sequences{j});
% 	end
% end
% sym_scores
% seqalign(sequences{1}, sequences{2})

%% 

% trips = 10;
simplified{trips} = [];
for i = 1:trips
	tmp = [];
	for j = 1:5:length(sequences{i}) - 4
		nt = sum(sequences{i}(j:j+4));
		if nt > 0
			tmp = [tmp, 'A'];
		else if nt < 0
				tmp = [tmp, 'C'];
			else
				tmp = [tmp, 'G'];
			end
		end			
	end
	simplified{i} = tmp;
end

%%

sym_scores = [];
for i = 1:trips
	for j = i:trips
		[score, alignment] = nwalign(simplified{i}, sequencesdriver1{j}, 'Alphabet', 'NT');
		sym_scores(i, j) = score;
		sym_scores(j, i) = score;
	end
end

% for i = 1:trips
% 	for j = i + 1:trips
% 		sym_scores(i, j) = seqalign(sequences{i}, sequences{j});
% 	end
% end
scores_lower = tril(sym_scores, -1);
scores_upper = triu(sym_scores,  1);
sym_scores = scores_lower(:, 1:end - 1) + scores_upper(:, 2:end)

%%

sorted = sort(sym_scores, 'descend');
probs = mean(sorted(1:5, :)) / max(mean(sorted(1:5, :)));
% hist(probs, 20)

hist(histfit(probs), 20)

