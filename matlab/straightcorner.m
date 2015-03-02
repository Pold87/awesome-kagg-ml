addpath('~/drivers/3');

trip = csvread([num2str(2) '.csv'], 1, 0);
trip = rotate(trip);
straight = zeros(size(trip,1),1);

threshold = .1;

prevdxdy = 0;


for iter = 2 : size(trip,1)
    tmp = trip(iter-1,:) - trip(iter,:);
    dxdy = tmp(1) / tmp(2);
    straight(iter) = abs(dxdy - prevdxdy) < threshold;
    prevdxdy = dxdy;
end

figure;
hold on;

colors = 'rk';

for iter = 0 : 1
    xs = trip(straight == iter, 1);
    ys = trip(straight == iter, 2);
    
    plot(xs, ys, ['.' colors(iter+1)]);
end