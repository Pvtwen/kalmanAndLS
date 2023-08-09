%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function main3_3_2   最小二乘 用当前时刻的先验估计
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=100; %仿真时间，时间序列总数
vector=zeros(1000,100); % 位移的误差
vector1=zeros(1000,100); % 速度的误差
zero1=[0 ;0];
zero2=[0 0];
for kk=1:1000
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
    Xls=zeros(2,N);% Kalman估计状态初始化

    Xls(:,1)=X(:,1)+sqrt(P0)*randn(2,1); % 在初始化的时候加入误差方差估计值 从而保持吻合
    
    err_P=zeros(N,2);
    err_P(1,1)=P0(1,1);
    err_P(1,2)=P0(2,2);
    I=eye(2);      % 二维系统

    % 第一列
    vector(kk,1)=Xls(1,1)-X(1,1);
    vector1(kk,1)=Xls(2,1)-X(2,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for k=2:N
        %物体下落，受状态方程的驱动
        X(:,k)=A*X(:,k-1)+B*U+W(k);
        
        % 位移传感器对目标进行观测
        Z(k)=H*X(:,k)+V(k);
        
        % Kalman滤波
        X_pre=A*Xls(:,k-1)+B*U; %状态预测
        P_pre=A*P0*A'+Q;  %协方差预测
        
        % 归一化后的各项系数
        Rt=[P_pre zeros(2,1);zeros(1,2) R];
        Sk=Rt^(0.5);
        Xmat=[eye(2);H];
        Xlinear=Sk^(-1)*Xmat;
        Ymat=[A*Xls(:,k-1);Z(k)];
        Ylinear=Sk^(-1)*Ymat;

        Kg=P_pre*H'*(H*P_pre*H'+R)^(-1); %计算Kalman增益
        %Xkf(:,k)=X_pre+Kg*(Z(k)-H*X_pre); % 状态更
        Xls(:,k)=(Xlinear'*Xlinear)^(-1)*Xlinear'*Ylinear; % 新的线性回归的估计值
        P0=(I-Kg*H)*P_pre*(I-Kg*H)'+Kg*R*Kg';%方差更新

        vector(kk,k)=Xls(1,k)-X(1,k);
        vector1(kk,k)=Xls(2,k)-X(2,k);
        % 误差均方值
        err_P(k,1)=P0(1,1);
        err_P(k,2)=P0(2,2);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
legend('通过最小二乘得到的实际位移误差方差','kalman估计方差');
xlabel('采样时间/s');
ylabel('方差');

figure
plot(Difference1,'-bo'); %实际的速度误差方差
hold on;
plot(errPx2,'-g+');
legend('通过最小二乘得到的实际速度误差方差','kalman估计方差');
xlabel('采样时间/s');
ylabel('方差');