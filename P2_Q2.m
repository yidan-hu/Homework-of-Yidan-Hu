clear all, close all

% Simulate ks system
dt=0.01; T=8; t=0:dt:T;

% Kuramoto-Sivashinsky equation (from Trefethen)
% u_t = -u*u_x - u_xx - nu*u_xxxx,  periodic BCs
%u = -sin(x)+2*cos(2*x)+3*cos(3*x)-4*sin(4*x);

KS = @(t,x)[0.5;(568*cos(2*x(1)))/87+(459*cos(3*x(1)))/29-(1472*sin(4*x(1)))/87-...
    (83*sin(x(1)))/87+(16*cos(4*x(1))+4*sin(2*x(1))+...
    9*sin(3*x(1))+cos(x(1)))*(2*cos(2*x(1))+3*cos(3*x(1))-...
    4*sin(4*x(1))-sin(x(1)))];     
options = odeset('RelTol',1e-10, 'AbsTol',1e-11); 

input=[]; output=[];
for j=1:100  % training trajectories
    x0=[0.2*rand;
        20*rand];
    [t,y] = ode45(KS,t,x0);
    input=[input; y(1:end-1,:)];
    output=[output; y(2:end,:)];
    plot(y(:,1),y(:,2)), hold on 
    plot(x0(1),x0(2),'ro')
end
grid on

%%
net = feedforwardnet([10 10 10]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net = train(net,input.',output.');


%%
figure(2) 
x0=(rand(2,1)-0.5);
[t,y] = ode45(KS,t,x0);
plot(y(:,1),y(:,2)), hold on
plot(x0(1),x0(2),'ro','Linewidth',[2])
grid on

ynn(1,:)=x0;
for jj=2:length(t)
    y0=net(x0);
    ynn(jj,:)=y0.'; x0=y0;
end
plot(ynn(:,1),ynn(:,2),':','Linewidth',[2])

figure(3)
subplot(2,2,1), plot(t,y(:,1),t,ynn(:,1),'Linewidth',[2])
subplot(2,2,3), plot(t,y(:,2),t,ynn(:,2),'Linewidth',[2])


figure(2)
x0=(rand(2,1)-0.5);
[t,y] = ode45(KS,t,x0);
plot(y(:,1),y(:,2)), hold on
plot(x0(1),x0(2),'ro','Linewidth',[2])
grid on

ynn(1,:)=x0;
for jj=2:length(t)
    y0=net(x0);
    ynn(jj,:)=y0.'; x0=y0;
end
plot(ynn(:,1),ynn(:,2),':','Linewidth',[2])

figure(3)
subplot(2,2,2), plot(t,y(:,1),t,ynn(:,1),'Linewidth',[2])
subplot(2,2,4), plot(t,y(:,2),t,ynn(:,2),'Linewidth',[2])

%%
figure(2)
figure(3)
subplot(2,2,1), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(2,2,2), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(2,2,3), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(2,2,4), set(gca,'Fontsize',[15],'Xlim',[0 8])

legend('KS','NN')