%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function main3_3_2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=100; %仿真时间，时间序列总数
vector=zeros(100,100); % 位移的误差
vector1=zeros(100,100); % 速度的误差
c=0.35;
for kk=1:100
    % 定义一个行向量存储每次实验每个时刻的方差
    
    % 噪声
    Q=[0,0;0,0]; % 过程噪声方差为0，即下落过程忽略空气阻力
    R=1;         % 观测噪声方差
    W=sqrt(Q)*randn(2,N);% 既然Q为0，则W=0；在此写出，方便对照理解
    V=sqrt(R)*randn(1,N);% 测量噪声V(k)
    % 系统矩阵
    A=[1,1;0,1]; %状态转移矩阵
    B=[0.5;1];   %控制量
    U=-1;
    H=[1,0];     %观测矩阵
    % 初始化
    X=zeros(2,N);  % 物体真实状态
    X(:,1)=[95;1]; % 初始位移和速度
    P0=[10,0;0,1]; % 初始误差

    Z=zeros(1,N);  
    Z(1)=H*X(:,1); % 初始观测值                                                                                 
    Xirls=zeros(2,N);% Kalman估计状态初始化

    Xirls(:,1)=X(:,1)+sqrt(P0)*randn(2,1); % 在初始化的时候加入误差方差估计值 从而保持吻合
    err_P=zeros(N,2);
    err_P(1,1)=P0(1,1);
    err_P(1,2)=P0(2,2);
    I=eye(2);      % 二维系统

    % 第一列
    vector(kk,1)=Xirls(1,1)-X(1,1);
    vector1(kk,1)=Xirls(2,1)-X(2,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for k=2:N
        %物体下落，受状态方程的驱动
        X(:,k)=A*X(:,k-1)+B*U+W(k);
        
        % 位移传感器对目标进行观测
        Z(k)=H*X(:,k)+V(k);
        if k>50 && k<60
            Z(k)=100;
        end
        % 1 状态预测
        Xirls_pre=A*Xirls(:,k-1)+B*U;
        P_pre=A*P0*A'+Q;

        Lt=[R zeros(1,2);zeros(2,1) P_pre]; % 归一化之前的线性回归的误差协方差矩阵 Lt=Lk*Lk'
        % 归一化之后的线性回归的各项系数
        Lk=Lt^(0.5);
        yk=Lk^(-1)*[Z(k);Xirls_pre];
        Ak=Lk^(-1)*[H;eye(2)];

        Kg=P_pre*H'*(H*P_pre*H'+R)^(-1); %计算Kalman增益

        % 2 求误差ri 用先验估计值  ri是一个(3x1)向量
        ri=yk-Ak*Xirls_pre;
        
        % 求Q
        Q1=zeros(3,3);
        % 在该循环中 求出ρ函数关于ri向量的导数
        for k1=1:3
            ri_row=ri(k1); % ri的第k行的值
            if ri_row==0  % 如果ri_row=0 将其设置为很小的一个数
                ri_row=0.001;
            end
            
            roul_der=roul_derive(ri_row,c); % 求导
            Q1(k1,k1)=roul_der*(1/ri_row);  % 设置Q1对角矩阵对角线上的值
        end
        
        % 求出估计值
        Xirls_after=(Ak'*Q1*Ak)^(-1)*Ak'*Q1*yk;
        % 因为有两个状态变量（位移和速度），分别设置两个阈值，只有当两个状态变量k时刻和k-1时刻的值的差值大于阈值，不断迭代
        while Xirls_after(1)-Xirls_pre(1)>1 && Xirls_after(2)-Xirls_pre(2)>0.1                      
            Xirls_pre=Xirls_after; % save
            % 与进入while循环之前相同
            ri=yk-Ak*Xirls_after;
            
            for k1=1:3
                ri_row=ri(k1);
                if ri_row==0
                    ri_row=0.001;
                end
            
                roul_der=roul_derive(ri_row,c);
                Q1(k1,k1)=roul_der*(1/ri_row);
            end
            Xirls_after=(Ak'*Q1*Ak)^(-1)*Ak'*Q1*yk;
        end
        
        P0=(I-Kg*H)*P_pre*(I-Kg*H)'+Kg*R*Kg';%方差更新
        Xirls(:,k)=Xirls_after; % 将该时刻的最后一次迭代值存入Xirls矩阵
        vector(kk,k)=Xirls(1,k)-X(1,k); % 第kk次实验的第k个时刻 位移的误差
        vector1(kk,k)=Xirls(2,k)-X(2,k); % 第kk次实验的第k个时刻 速度的误差
        % 误差均方值
        err_P(k,1)=P0(1,1);
        err_P(k,2)=P0(2,2);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

x_hat(:,1)=Xirls(:,1);
P(:,:,1)=[10,0;0,1];
for k=2:N
    xp=A*x_hat(:,k-1)+B*U;
    Pp=A*P(:,:,k-1)*A'+Q;
    K=Pp*H'*(H*Pp*H'+R)^(-1);
    x_hat(:,k)=xp+K*(Z(k)-H*xp);
    P(:,:,k)=(eye(2)-K*H)*Pp;
    
end
% 计算vector矩阵中 每一列的方差 存储在行向量中
Difference=var(vector,0,1);
Difference1=var(vector1,0,1);
errPx=transpose(err_P);
errPx1=errPx(1,:);
errPx2=errPx(2,:);
% 将err_P(k,1)和Difference()画在同一张图中，比较卡尔曼滤波估计出来的Pk与实际的方差相差多少.
figure
plot(Difference,'-bo'); % 实际的位移误差方差
hold on;
plot(errPx1,'-g+');
legend('实际位移误差方差','kalman估计方差');
xlabel('采样时间/s');
ylabel('方差');

figure
plot(Difference1,'-bo'); %实际的速度误差方差
hold on;
plot(errPx2,'-g+');
legend('实际速度误差方差','kalman估计方差');
xlabel('采样时间/s');
ylabel('方差');
figure(3);
hold on;
plot(X(1,:));
plot(Xirls(1,:),'--');
plot(x_hat(1,:),'g--');
legend('true state','IRLS estimate','KF estimate');

figure(4);
hold;
plot(X(2,:));
plot(Xirls(2,:),'--');
plot(x_hat(2,:),'g--');
legend('true state','IRLS estimate','KF estimate');
function roul_der = roul_derive(ri,c)
    if abs(ri)<c
        roul_der=ri;
    elseif ri>0
        roul_der=c;
    else
        roul_der=-c;
    end
end