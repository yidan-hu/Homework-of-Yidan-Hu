%% Q1.4 Find sparse regression using SINDy and interpolation
clear all; close all; clc;
x1 = [20,20,52,83,64,68,83,12,36,150,110,60,7,10,70,...
    100,92,70,10,11,137,137,18,22,52,83,18,10,9,65];
x2 = [32,50,12,10,13,36,15,12,6,6,65,70,40,9,20,...
    34,45,40,15,15,60,80,26,18,37,50,35,12,12,25];
slices = 30;
t = linspace(0,58,slices);
dt = t(2) - t(1);
r = 2;
dt_new = 0.5;
t_new = 0:dt_new:58;
n = length(t_new);
%
x1_interp = interp1(t,x1,t_new,'spline').';
x2_interp = interp1(t,x2,t_new,'spline').';
X =[x1_interp(1:end-1) x2_interp(1:end-1)];

for j=1:n-1
    x1dot(j) = (x1_interp(j+1)-x1_interp(j))/(dt_new);
    x2dot(j) = (x2_interp(j+1)-x2_interp(j))/(dt_new);
end
dx = [x1dot.' x2dot.'];
% SINDy
polyorder = 3;
Theta = poolData(X,r,polyorder);
m =size(Theta,2);
lambda = 0.02;
Xi = sparsifyDynamics(Theta,dx,lambda,r)
poolDataLIST({'x','y'},Xi,r,polyorder);
% reconstraction of data with sparse regression
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,2));
 Beta(1)= Xi(1,1);
 Beta(2)= Xi(2,1);
 Beta(3)= Xi(1,2);
 Beta(4)= Xi(2,2);
Beta(5)= Xi(2,2);
Beta(6)= Xi(3,2);
[tt,xx]=ode45(@(tt,xx) LV2(tt,xx,Beta),t(1:end-1),[20 32],options);

%
figure(1)
subplot(2,1,1)
plot(t+1845,x1,'ro-',tt+1845,xx(:,1),'r*-')
legend('Lynx','SINDy Lynx');xlim([1845 1907]);ylim([-25 240]);
grid on

subplot(2,1,2)
plot(t+1845,x2,'bo-',tt+1845,xx(:,2),'b*-')
legend('Hare','SINDy Hare');xlim([1845 1907]);ylim([0 120]);
grid on

%% bagging sparse regression SINDy
clear all; clc;

x1 = [20,20,52,83,64,68,83,12,36,150,110,60,7,10,70,...
    100,92,70,10,11,137,137,18,22,52,83,18,10,9,65];
x2 = [32,50,12,10,13,36,15,12,6,6,65,70,40,9,20,...
    34,45,40,15,15,60,80,26,18,37,50,35,12,12,25];
slices = 30;
t = linspace(0,58,slices);
dt = t(2) - t(1);
r = 2;
dt_new = 0.5;
%t_new = 0:dt_new:58;

m = 50;
X_bagging = zeros(29,2);
for j=1:m
    nn=26; 
    k = randi([1 31-nn]); 
    x1_random = zeros(1,nn);
    x2_random = zeros(1,nn);
    ts_random = zeros(1,nn);
    
    % define the new data library
    for i=1:nn
        t_random(i) = t(k-1+i);
        x1_random(i) = x1(k-1+i);
        x2_random(i) = x2(k-1+i);
    end
    x_random = [x1_random; x2_random];
    
    t_new = t_random(1):dt_new:t_random(24);
    n = length(t_new);

    %
    x1_interp = interp1(t_random,x1_random,t_new,'spline').';
    x2_interp = interp1(t_random,x2_random,t_new,'spline').';
    X =[x1_interp(1:end-1) x2_interp(1:end-1)];

    for j=1:n-1
        x1dot(j) = (x1_interp(j+1)-x1_interp(j))/(dt_new);
        x2dot(j) = (x2_interp(j+1)-x2_interp(j))/(dt_new);
    end
    dx = [x1dot.' x2dot.'];

    % SINDy
    polyorder = 3;
    Theta = poolData(X,r,polyorder);
    m =size(Theta,2);
    lambda = 0.025;
    Xi = sparsifyDynamics(Theta,dx,lambda,r)
    poolDataLIST({'x','y'},Xi,r,polyorder);

    % reconstraction of data with sparse regression
    options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,2));
    Beta(1)= Xi(1,1);
    Beta(2)= Xi(2,1);
    Beta(3)= Xi(1,2);
    Beta(4)= Xi(2,2);
    Beta(5)= Xi(2,2);
    Beta(6)= Xi(3,2);
    [tt,xx]=ode45(@(tt,xx) LV2(tt,xx,Beta),t(1:end-1),[20 32],options);
    X_bagging = X_bagging + 1/m*xx;
end

figure(2)
subplot(2,1,1)
plot(t+1845,x1,'ro-',tt+1845,X_bagging(:,1),'r*-')
legend('Lynx','Bagging SINDy Lynx');xlim([1845 1907]);ylim([-25 240]);
grid on

subplot(2,1,2)
plot(t+1845,x2,'bd-',tt+1845,X_bagging(:,2),'b*-')
legend('Hare','Bagging SINDy Hare');xlim([1845 1907]);ylim([0 120]);
grid on


%% 
function yout = poolData(yin,nVars,polyorder)
n = size(yin,1); 

ind = 1;
% poly order 0
yout(:,ind) = ones(n,1);
ind = ind+1;

%poly order 1
for i=1:nVars
    yout(:,ind) = yin(:,i);
    ind = ind+1;
end

if(polyorder>=2)    % poly order 2
    for i=1:nVars
        for j=i:nVars
            yout(:,ind) = yin(:,i).*yin(:,j);
            ind = ind+1;
        end
    end
end
if(polyorder>=3)    % poly order 3
    for i=1:nVars
        for j=i:nVars
            for k=j:nVars
                yout(:,ind) = yin(:,i).*yin(:,j).*yin(:,k);
                ind = ind+1;
            end
        end
    end
end
end

%----------------------------------------------------------
function Xi = sparsifyDynamics(Theta,dXdt,lambda,n)

Xi = Theta\dXdt;  % initial guess: Least-squares

% lambda is our sparsification knob.
for k=1:10
    smallinds = (abs(Xi)<lambda);   % find small coefficients
    Xi(smallinds)=0;                % and threshold
    for ind = 1:n                   % n is state dimension
        biginds = ~smallinds(:,ind);
        % Regress dynamics onto remaining terms to find sparse Xi
        Xi(biginds,ind) = Theta(:,biginds)\dXdt(:,ind); 
    end
end
end

%----------------------------------------------------------
function dx = LV2(t,x,Beta)
dx = [
% Beta(1)*x(1)+Beta(2)*x(2);
% Beta(3)*x(1)+Beta(4)*x(2);
Beta(1)+Beta(2)*x(1)+Beta(3)*x(2);
Beta(4)+Beta(5)*x(1)+Beta(6)*x(2);
];
end

%----------------------------------------------------------
function yout = poolDataLIST(yin,ahat,nVars,polyorder)
n = size(yin,1);

ind = 1;
% poly order 0
yout{ind,1} = ['1'];
ind = ind+1;

% poly order 1
for i=1:nVars
    yout(ind,1) = yin(i);
    ind = ind+1;
end

if(polyorder>=2)
    % poly order 2
    for i=1:nVars
        for j=i:nVars
            yout{ind,1} = [yin{i},yin{j}];
            ind = ind+1;
        end
    end
end

if(polyorder>=3)
    % poly order 3
    for i=1:nVars
        for j=i:nVars
            for k=j:nVars
                yout{ind,1} = [yin{i},yin{j},yin{k}];
                ind = ind+1;
            end
        end
    end
end

output = yout;
newout(1) = {''};
for k=1:length(yin)
    newout{1,1+k} = [yin{k},'dot'];
end
% newout = {'','xdot','ydot','udot'};
for k=1:size(ahat,1)
    newout(k+1,1) = output(k);
    for j=1:length(yin)
        newout{k+1,1+j} = ahat(k,j);
    end
end
newout
end