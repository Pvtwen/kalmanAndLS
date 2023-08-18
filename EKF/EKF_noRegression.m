%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function main3_3_2
% non-linear model， No OLS, No IRLS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=100; % simulation time
T=0.1;

kx=0.01;ky=0.05;g=9.8;
 
for i=1:4
    vector_matrix(:,:,i)=zeros(100,N);
end

%  do kk time experiments
for kk=1:100
    delta_w=1e-3; 
    Q=diag([0.5,1,0.5,1]);          % system process noise covariance matrix 
    R=diag([1,1]);         % observation noise variance matrix,  observation states, distance, angle
    
    % initialization
    X=zeros(4,N);  
    
    % system noise and observation noise
    W=sqrt(Q)*randn(4,N);
    V=sqrt(R)*randn(2,N);

    X(:,1)=[0,50,500,0]; % initial x position=0,velocity x,y position, velocity y;
    P0=eye(4);  
    
    Z=zeros(2,N);  
    for t=2:N
        x1=X(1,t-1)+X(2,t-1)*T+W(1,t);
        v1=X(2,t-1)-(kx*X(2,t-1)^(2))*T+W(2,t);
        y1=X(3,t-1)-X(4,t-1)*T+W(3,t);
        v2=X(4,t-1)-(ky*X(4,t-1)^(2)-g)*T+W(4,t);
        X(:,t)=[x1;v1;y1;v2];
    end
    
     % init observation   
    for t=1:N
        x1=X(1,t);y1=X(3,t);
        r1=Dist(x1,y1)+V(1,t);
        alpha1=atan(x1/y1)*180/pi+V(2,t);                                                                                                                                                                                                                                                                   
        Z(:,t)=[r1;alpha1];
    end

    X_ekf=zeros(4,N);% init irls estimation

    X_ekf(:,1)=X(:,1)+sqrt(P0)*randn(4,1); % Introducing an estimated error variance during initialization to maintain alignment
    err_P=zeros(4,N); % Each column represents the four values of the Kalman's P matrix at that moment, with each row recording each timestamp.
    for k2=1:4
        err_P(k2,1)=P0(k2,k2);
    end
    I=eye(4);      % 4 dimensions systems

    % real error estimation
    for j=1:4
        vector=vector_matrix(:,:,j);
        vector(kk,1)=X_ekf(j,1)-X(j,1);
        vector_matrix(:,:,j)=vector;
    end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for k=2:N
        % state prediction, apply the state prediction to the variable Xekf)
        % xpre_hat{k}=f(,,0)
        x1=X_ekf(1,k-1)+X_ekf(2,k-1)*T;
        v1=X_ekf(2,k-1)-kx*X_ekf(2,k-1)^(2)*T;
        y1=X_ekf(3,k-1)-X_ekf(4,k-1)*T;
        v2=X_ekf(4,k-1)-(ky*X_ekf(4,k-1)^(2)-g)*T;
        Xekf_pre=[x1;v1;y1;v2];
        %Xekf_pre=F*X_ekf(:,k-1);
        
        % observation prediction
        r=Dist(x1,y1);alpha2=atan(x1/y1)*180/pi;
        Zekf_pre=[r;alpha2];

        % introducing A(k) matrix: partial derivative
        A=[1 T 0 0;0 1-2*kx*X_ekf(2,k-1)*T 0 0;0 0 1 -T;0 0 0 1-2*ky*X_ekf(4,k-1)*T];
        dd=Dist(x1,y1); de=1+(x1/y1)^(2);
        H=[x1/dd 0 y1/dd 0;(1/y1)/de 0 (-x1/y1^(2))/de 0]; % jacobi matrix

        % pre of the P
        P_pre=A*P0*A'+Q;

        K=P_pre*H'*(H*P_pre*H'+R)^(-1);
  
        X_ekf(:,k)=Xekf_pre+K*(Z(:,k)-Zekf_pre);

        P0=(I-K*H)*P_pre;

        % update error
        for j=1:4
            vector=vector_matrix(:,:,j);
            vector(kk,k)=X_ekf(j,k)-X(j,k);
            vector_matrix(:,:,j)=vector;
        end

        for k2=1:4
            err_P(k2,k)=P0(k2,k2);
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

% calculate every column of the vector matrix,
% store the value in the row vector
for i=1:4
    vector=vector_matrix(:,:,i);
    Difference_matrix(:,:,i)=var(vector,0,1);
end

for i=1:4
    errPx_matrix(:,:,i)=err_P(i,:);
end
figure(1)
hold on;box on;
plot(X(1,:),X(3,:),'-k.');
plot(X_ekf(1,:),X_ekf(3,:),'-r+');
plot(Z(1,:).*sin(Z(2,:)*pi/180),Z(1,:).*cos(Z(2,:)*pi/180),'+');
legend('真实轨迹','EKF轨迹','观测轨迹');

figure(2)
plot(Difference_matrix(:,:,1),'-bo');
hold on;
plot(errPx_matrix(:,:,1),'-g+');
legend('real X displacement error variance','ekf estimation error variance');
xlabel('sampling time/s');
ylabel('variance');

figure(3)
plot(Difference_matrix(:,:,2),'-bo');
hold on;
plot(errPx_matrix(:,:,2),'-g+');
legend('real X velocity error variance','ekf estimation error variance');
xlabel('sampling time/s');
ylabel('variance');

figure(4)
plot(Difference_matrix(:,:,3),'-bo');
hold on;
plot(errPx_matrix(:,:,3),'-g+');
legend('real Y displacement error variance','ekf estimation error variance');
xlabel('sampling time/s');
ylabel('variance');

figure(5)
plot(Difference_matrix(:,:,4),'-bo');
hold on;
plot(errPx_matrix(:,:,4),'-g+');
legend('real Y velocity error variance','ekf estimation error variance');
xlabel('sampling time/s');
ylabel('variance');

function d=Dist(X1,X2)
    d=sqrt(X1^(2)+X2^(2));
end
