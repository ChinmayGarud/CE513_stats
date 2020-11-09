clear all
close all
clc
% Data
global x sample_mean N sig
c=[10.3;12.2;8.5;14.2];
phi=[34.2;31.1;35.7;30.8];
x=[c,phi];
sample_mean=mean(x);
N=height(x);

% Known covariance
sig=[16,0;0,9];

% Assumed prior 1
mu_prior1=[12;28];
sig_prior1=[9,0;0,8];

% % Posterior from prior 1
% [sig_post1,mu_post1]=posterior_calc(sig_prior1,mu_prior1);
% % Posterior from prior 2
% [sig_post2,mu_post2]=posterior_calc(sig_prior2,mu_prior2);

% Define the intervals for the mu parametrized by theta for both dimensions
% of x
a1=0; a2=30; delta1=0.2; theta1=a1:delta1:a2;
b1=20; b2=40; delta2=0.2; theta2=b1:delta2:b2;

% Joint dist
for i=1:length(theta1)
    for j=1:length(theta2)
        theta=[theta1(i),theta2(j)];
        Mu=theta;
        like=mvnpdf(x,Mu,sig);
        prior=mvnpdf(Mu,transpose(mu_prior1),sig_prior1);
        q(i,j)=prod(like)*prior;
    end
end
f12=q/sum(sum(q))/delta1/delta2;

% Marginal distribution
for i=1:length(theta1)
 f1(i)=sum(sum(q(i,:)));
end
f1=f1/sum(f1*delta1);

% Marginal distribution 2
for k=1:length(theta2)
 f2(k)=sum(sum(q(:,k)));
end
f2=f2/sum(f2*delta2);

% Plots of marginal densities
plot(f1)
hold on
plot(f2)
hold off
% Plot of joint density
figure
surf(f12)

% Assumed prior 2
mu_prior2=[14;32];
sig_prior2=[10,0;0,12];

% Joint dist
for i=1:length(theta1)
    for j=1:length(theta2)
        theta=[theta1(i),theta2(j)];
        Mu=theta;
        like=mvnpdf(x,Mu,sig);
        prior=mvnpdf(Mu,transpose(mu_prior2),sig_prior2);
        q(i,j)=prod(like)*prior;
    end
end
f12=q/sum(sum(q))/delta1/delta2;

% Marginal distribution
for i=1:length(theta1)
 f1(i)=sum(sum(q(i,:)));
end
f1=f1/sum(f1*delta1);

% Marginal distribution 2
for k=1:length(theta2)
 f2(k)=sum(sum(q(:,k)));
end
f2=f2/sum(f2*delta2);


% plots
figure
plot(f1)
hold on
plot(f2)

figure
surf(f12)

function [sig_post,mu_post]=posterior_calc(sig_prior,mu_prior)
    global x sample_mean N sig
    sig_post=inv(inv(sig_prior)+(N*inv(sig)));
    mu_post=(inv(inv(sig_prior)+(N*inv(sig))))*((inv(sig_prior)*mu_prior)+(N*inv(sig)*transpose(sample_mean)));
end




