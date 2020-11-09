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
global delta1 delta2 theta1 theta2
a1=0; a2=30; delta1=0.2; theta1=a1:delta1:a2;
b1=20; b2=40; delta2=0.2; theta2=b1:delta2:b2;

q=joint_posterior(sig_prior1,mu_prior1)
f12=q/sum(sum(q))/delta1/delta2;
f1=marg_stats1(q);
f2=marg_stats2(q)


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

function q=joint_posterior(sig_prior,mu_prior)
    global theta1 theta2 x sig
    % Joint dist
    for i=1:length(theta1)
        for j=1:length(theta2)
            theta=[theta1(i),theta2(j)];
            Mu=theta;
            like=mvnpdf(x,Mu,sig);
            prior=mvnpdf(Mu,transpose(mu_prior),sig_prior);
            q(i,j)=prod(like)*prior;
        end
    end
end

function f1=marg_stats1(q)
    global theta1 delta1
    % Marginal distribution
    for i=1:length(theta1)
        f1(i)=sum(sum(q(i,:)));
    end
    f1=f1/sum(f1*delta1);
end

function f2=marg_stats2(q)
    global theta2 delta2
    % Marginal distribution
    for i=1:length(theta2)
        f2(i)=sum(sum(q(:,i)));
    end
    f2=f2/sum(f2*delta2);
end

