clear all;clc
format long
% the data
slices = 30;
tspan=linspace(0,58,30);
ye=[20,20,52,83,64,68,83,12,36,150,110,60,7,10,70,...
    100,92,70,10,11,137,137,18,22,52,83,18,10,9,65]';
ze=[32,50,12,10,13,36,15,12,6,6,65,70,40,9,20,...
    34,45,40,15,15,60,80,26,18,37,50,35,12,12,25]';
yy=[ye ze];

% coefficient beta0=[b p r d];
beta0=[5 0.3 0.1 0.3]; 
y0=[20; 32];
lb=[0.1 0.1 0.1 0.1];
ub=[15 15 15 15];   
yexp=yy(2:end,:);
k0=beta0;
[k,resnorm,~,~,~,~,~] = ...
    lsqnonlin(@ObjFunc,k0,lb,ub,[],tspan,y0,yexp);
fprintf('\n\n Solving problem using lsqnonlin(),the fit values is :\n')
fprintf('\t b = %.6f\n',k(1))
fprintf('\t p = %.6f\n',k(2))
fprintf('\t r = %.6f\n',k(3))
fprintf('\t d = %.6f\n',k(4))

ts=0:2:max(tspan);
% 
[ts, ys]=ode45(@LotkaVolterra,ts,y0,[],k);
[~, XXsim] = ode45(@LotkaVolterra,tspan,y0,[],k);
y=XXsim(2:end,:);


figure(1);
subplot(2,1,1)
plot(tspan+1845,yy(:,1),'ro-',ts+1845,ys(:,1),'r*--');
legend('Hare','fit curve Hare');
xlabel('Year');xlim([1845 1905]);
ylabel('Populations');ylim([0 155]);
grid on
subplot(2,1,2)
plot(tspan+1845,yy(:,2),'bd-',ts+1845,ys(:,2),'b*--');
legend('Lynx','fit curve Lynx');
xlabel('Year');xlim([1845 1905]);ylim([0 85]);
ylabel('Populations');
grid on

%---------------------------------------------------------
function f = ObjFunc(k,tspan,y0,yexp)           
[~, Xsim] = ode45(@LotkaVolterra,tspan,y0,[],k) ;
ysim = Xsim(2:end,:);
size(ysim);
size(yexp);
f=ysim-yexp;
end
%----------------------------------------------------------

function dydt = LotkaVolterra(~,y,k)
beta(1)=k(1);
beta(2)=k(2);
beta(3)=k(3);
beta(4)=k(4);
dydt = zeros(2,1);
sxy = y(1)*y(2);
dydt(1) = k(1)*y(1)-k(2)*sxy;
dydt(2) = k(3)*sxy-k(4)*y(2);
end