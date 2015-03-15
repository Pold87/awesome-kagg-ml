folder = '../drivers/2/';

trips = 200;
sequences{trips} = [];

turns = 50;
freqs = 20;
bins = 10;

% cntr = 1;
%%% matlabpool to initialize workers
parfor i = 1:trips
	trip = csvread([folder num2str(i) '.csv'], 1, 0);
	% trip2 = csvread([num2str(2) '.csv'], 1, 0);

% 	trip = trip(1:50,:);
% 	figure;
% 	trip = rotate(trip);
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
1
% trips = 10;
simplified{trips} = [];
nts = [];
threshold = .5;
for i = 1:trips
	tmp = [];
	for j = 1:5:length(sequences{i}) - 4
		nt = sum(sequences{i}(j:j+4));
        nts = [nts ; nt];
		if nt > threshold
			tmp = [tmp, 'A'];
        elseif nt < -threshold
            tmp = [tmp, 'C'];
        else
            tmp = [tmp, 'G'];
		
		end			
	end
	simplified{i} = tmp;
end

%%
2
sym_scores = repmat([-100000], trips, trips);
for i = 1:trips
	for j = i+1:trips
		[score, alignment] = bestalignment(simplified{i},simplified{j});
        nwalign(simplified{i}, simplified{j}, 'Alphabet', 'NT', 'ScoringMatrix', 'NUC44', 'GapOpen', 10000);
		sym_scores(i, j) = score;
		sym_scores(j, i) = score;
	end
end

% for i = 1:trips
% 	for j = i + 1:trips
% 		sym_scores(i, j) = seqalign(sequences{i}, sequences{j});
% 	end
% end
% scores_lower = tril(sym_scores, -1);
% scores_upper = triu(sym_scores,  1);
% sym_scores = scores_lower(:, 1:end - 1) + scores_upper(:, 2:end);

%%
3
sorted = sort(sym_scores, 'descend');
probs = mean(sorted(1:5, :)) / max(mean(sorted(1:5, :)));
size( sym_scores(sym_scores > 0))
% hist(probs, 20)
% HeatMap(flipdim(sym_scores,1));

for i = 1 : 50
    [val, idx] = max(sym_scores(i,:));
    trip1 = rotate(csvread([folder num2str(i) '.csv'], 1, 0));
    trip2 = rotate(csvread([folder num2str(idx) '.csv'], 1, 0));

    figure;
    hold on
    title(['symscore= ' num2str(val) ' | ' num2str(i) ' and ' num2str(idx)]);
    plot(trip1(:,1), trip1(:,2), '-k');
    plot(trip2(:,1), trip2(:,2), '-r');
    hold off
end

for i = 1:200; 
    for j = 1:200; 
        if sym_scores(i,j) > 50  ; 
            [num2str(i) ',' num2str(j)]
            trip1 = rotate(csvread([folder num2str(i) '.csv'], 1, 0));
            trip2 = rotate(csvread([folder num2str(idx) '.csv'], 1, 0));

            figure;
            hold on
            title(['symscore=' num2str(sym_scores(i,j))]);
            plot(trip1(:,1), trip1(:,2), '-k');
            plot(trip2(:,1), trip2(:,2), '-r');
            hold off
        end
    end
end


