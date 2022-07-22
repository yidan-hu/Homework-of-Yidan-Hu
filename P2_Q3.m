% Reaction-diffusion system SVD
clear all, close all
%%% load data from original provided code
load('reaction_diffusion_big.mat')
%%% plot both u and v at last snapshot

%%% apply SVD on u, for every snapshot
dt = t(2) - t(1);
numt = length(t);
U = zeros(length(x),length(y),length(t));
V = zeros(length(x),length(y),length(t));
S = zeros(length(x),length(y),length(t));
for j=1:numt
    [U(:,:,j),S(:,:,j),V(:,:,j)] = svd(u(:,:,j),'econ');
end
%%% plot sigma of SVD for each snapshot
%%% It could be found that the previous 10 space could be a good approx.
for jj=1:numt
   semilogx(abs(diag(S(:,:,jj))),'ko-'), hold on
   title('Sigma of every snapshot')
end

%%% truncate to rank-r, in order to represent the orginal model in low rank
%%% space
r = 10;
U_r = U(:,1:r,:);
S_r = S(1:r,1:r,:);
V_r = V(:,1:r,:);

%%% Convert approximated u with low rank space
u_approx = zeros(length(x),length(y),numt);
for jj=1:numt
    u_approx(:,:,jj) = U_r(:,:,jj)*S_r(:,:,jj)*V_r(:,:,jj)';
end

%%% compare the orginal u and the approximated u
figure;
subplot(2,1,1)
pcolor(x,y,u_approx(:,:,end)); shading interp; colormap(hot)
title('approximated u')

subplot(2,1,2)
pcolor(x,y,u(:,:,end)); shading interp; colormap(hot)
title('orginal u')

%%% compress all data together and find low-rank space
[k1,k2,k3] = size(u)
uu = reshape(u,k1*k2,k3);
[UU SS VV] = svd(uu,'econ');

figure;
semilogx(abs(diag(SS)),'ko-')
title('Sigma of all data')
rr = 4;
UU_r = UU(:,1:rr);
SS_r = SS(1:rr,1:rr);
VV_r = VV(:,1:rr);
uu_approx = UU_r*SS_r*VV_r';
u_approx2 = reshape(uu_approx,k1,k2,k3);

figure;
subplot(2,1,1)
pcolor(x,y,u_approx2(:,:,end)); shading interp; colormap(hot)
title('approximated u')

subplot(2,1,2)
pcolor(x,y,u(:,:,end)); shading interp; colormap(hot)
title('orginal u')


