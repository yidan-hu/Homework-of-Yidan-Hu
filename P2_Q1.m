%% Train a NN for KS equation
clear all; close all; clc

N=4;
%N=16;
dt=0.01; T=50; ts=0:dt:T; 


%% Networks
input=[];
output=[];
for j=1:100
u0 = randn(N,1);
[tx, xx, ux] = KSx(u0,N,ts);
input = [input; ux(1:end-1,:)];
output = [output; ux(2:end,:)];
end

net = feedforwardnet([10 10 10]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net = train(net,input.',output.');

u1 = randn(N,1);
[tx,xx,ux] = KSx(u1,N,ts);
x0 = ux(1,:).';
ynn(1,:)=x0;
for jj=2:length(tx)
    y0=net(x0);
    ynn(jj,:)=y0.'; 
    x0=y0;
end

figure(1)
contour(xx,tx,ux),shading interp, colormap("hot")

figure(2)
contour(xx,tx,ynn),shading interp, colormap("hot")

figure(3)
surf(ynn-ux)
title('error')
colormap parula
colorbar



%%
function [tsave, xsave, usave] = KSx(u,N,ts)
%N=64;
%dt=0.01; T=140; ts=0:dt:T; 

x = 2*pi*(1:N)'/N; 
v = fft(u); 
nu = 0.05;

% % % % % %
%Spatial grid and initial condition:
h = ts(2)-ts(1); 
k = [0:N/2-1 0 -N/2+1:-1]';  
L = k.^2 - nu*k.^4; 
E = exp(h*L); E2 = exp(h*L/2);
M = 16;
r = exp(1i*pi*((1:M)-.5)/M); 
LR = h*L(:,ones(M,1)) + r(ones(N,1),:); 
Q = h*real(mean( (exp(LR/2)-1)./LR ,2));
f1 = h*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 , 2) );
f2 = h*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
f3 = h*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));

% Main time-stepping loop:
uu = u; tt = 0;
%get period in time
tmax = max(ts); 

nmax = round(tmax/h); nplt = floor((tmax/1000)/h); g = -0.5i*k;
tt = zeros(1,nmax);
uu = zeros(N,nmax);

for n = 1:nmax
    t = n*h;
    Nv = g.*fft(real(ifft(v)).^2);
    a = E2.*v + Q.*Nv;
    Na = g.*fft(real(ifft(a)).^2); 
    b = E2.*v + Q.*Na;
    Nb = g.*fft(real(ifft(b)).^2);
    c = E2.*a + Q.*(2*Nb-Nv);
    Nc = g.*fft(real(ifft(c)).^2);
    v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3;     
    if mod(n,nplt)==0
        n;
        u = real(ifft(v));
        uu(:,n) = u; 
        tt(n) = t;
    end
end

%
cutoff = tt > 0;
cutoff = cutoff & tt < 1;

tsave = tt(cutoff);
xsave = x/(2*pi);
usave = uu(:,cutoff).';

end