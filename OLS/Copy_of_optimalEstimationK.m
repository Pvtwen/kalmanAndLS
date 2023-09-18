%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function main3_3_2   最小二乘 用当前时刻的先验估计
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
N=100; %仿真时间，时间序列总数

    Q=[1,0;0,1]; % 过程噪声方差为0，即下落过程忽略空气阻力
    R=1;         % 观测噪声方差
    W=sqrt(Q)*randn(2,N);% 既然Q为0，则W=0；在此写出，方便对照理解
    V=sqrt(R)*randn(1,N);% 测量噪声V(k)
    % 系统矩阵
    A=[1,1;0,1]; %状态转移矩阵

    H=[1,0];     %
    % 
    X=zeros(2,N);  % 
    X(:,1)=[95;1]; % 
    P0=[10,0;0,1]; % 

    Z=zeros(1,N);  
    Z(1)=H*X(:,1); %                                                                              
    Xls=zeros(2,N);% 

    Xls(:,1)=X(:,1)+sqrt(P0)*randn(2,1); % 
    
    I=eye(2);      % 
    for k=2:N
        X(:,k)=A*X(:,k-1)+W(k);
        Z(k)=H*X(:,k)+V(k);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for k=2:N
        Xls_pre=A*Xls(:,k-1); % the prior_estimation of time k
        P_pre=Q+A*P0*A'; % the prior_covariance mattrix of time k

        % computing the Kalman gain
        K=P_pre*H'*(R+H*P_pre*H')^(-1);
        Xls(:,k)=Xls_pre+K*(Z(:,k)-H*Xls_pre); % computing the time k's X estimation
        P0=(I-K*H)*P_pre; % computing the time k covariance matrix
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure 
plot(Xls(1,:),'-bo');
hold on;
plot(X(1,:),'-ro');
legend('estimation','real');
xlabel('sampling time/s');
ylabel('value');