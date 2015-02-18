addpath('~/drivers/3');
max_iter = 10;
figure;
hold on;

for iter = 1 : max_iter
    trip = csvread([num2str(iter) '.csv'], 1, 0);
%     plot(trip(:,1), trip(:,2), '-k');
    
    trip = rotate(trip);
    plot(trip(:,1), trip(:,2), '-');
end

axis([-5000, 5000, -5000, 5000]);