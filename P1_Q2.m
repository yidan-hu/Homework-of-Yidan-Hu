%% time-delay DMD model
% Data x1--Snowshoe Hare   x2--Canada Lynx   ts--time
x1 = [20,20,52,83,64,68,83,12,36,150,110,60,7,10,70,...
    100,92,70,10,11,137,137,18,22,52,83,18,10,9,65];
x2 = [32,50,12,10,13,36,15,12,6,6,65,70,40,9,20,...
    34,45,40,15,15,60,80,26,18,37,50,35,12,12,25];
slices = 30;
ts = linspace(1845,1843+2*slices,slices);
dt = ts(2) - ts(1);

% Establish H
x =[];

kk = 5;
for j=1:kk
    x = [x;x1(j:30-kk+j);x2(j:30-kk+j)];  
end

% Target rank
r = 2*kk;

% optDMD 
t1 = linspace(1845,1843+2*(31-kk),31-kk);
imode = 1;

[w,e,b] = optdmd(x,t1,r,imode);

% reconstructed values
X = w*diag(b)*exp(e*ts);

% forecast population states of the next 10 years
% t---time
t = linspace(1905,1907,2);
X_future = w*diag(b)*exp(e*t);

figure(1);
subplot(2,1,1)
plot(ts,x1,'ro-',ts,X(1,:),'r*--',t,X_future(1,:),'rx--')
legend('Hare','Time-delay DMD Hare','Predicted Hare');
xlabel('Year');xlim([1845 1907]);
ylabel('Populations');ylim([0 200]);
grid on
subplot(2,1,2)
plot(ts,x2,'bd-',ts,X(2,:),'b*--',t,X_future(2,:),'bx--')
legend('Lynx','Time-delay DMD Lynx','Predicted Lynx');
xlabel('Year');xlim([1845 1907]);
ylabel('Populations');ylim([0 105]);
grid on

%% Bagging

m = 100;
kk = 3;
X_bagging = zeros(2*kk,30);
X_future_bagging = zeros(2*kk,2);
for j=1:m
    n=27; % 80%*30
    k = randi([1 31-n]); 
    x1_random = zeros(1,n);
    x2_random = zeros(1,n);
    ts_random = zeros(1,n);

    % define the new 60% data library
    for i=1:n
        ts_random(i) = ts(k-1+i);
        x1_random(i) = x1(k-1+i);
        x2_random(i) = x2(k-1+i);
    end
    
    % x_random = [x1_random; x2_random];
    x_random =[];
    
    for i=1:kk
        x_random = [x_random;x1_random(i:n-kk+i);x2_random(i:n-kk+i)];  
    end

    r = 2*kk;
    
    % optDMD 
    t1_random = linspace(ts(k),ts(k)+2*(n+1-kk),n+1-kk);
    imode = 1;
    [w_r,e_r,b_r] = optdmd(x_random,t1_random,r,imode);

    % reconstructed values
    X_random = w_r*diag(b_r)*exp(e_r*ts);
    X_bagging = X_bagging + 1/m*X_random;

    X_future_r = w_r*diag(b_r)*exp(e_r*t);
    X_future_bagging = X_future_bagging + 1/m*X_future_r;
end

figure(2);
subplot(2,1,1)
plot(ts,x1,'ro-',ts,X_bagging(1,:),'rp--',t,X_future_bagging(1,:),'rx--')
legend('Hare','bagging Hare','Predicted Hare');
xlabel('Year');xlim([1845 1907]);
ylabel('Populations');ylim([0 180]);
grid on
subplot(2,1,2)
plot(ts,x2,'bd-',ts,X_bagging(2,:),'bp--',t,X_future_bagging(2,:),'bx--')
legend('Lynx','bagging Lynx','Predicted Lynx');
xlabel('Year');ylabel('Populations');
xlim([1845 1907]);ylim([0 95]);
grid on

