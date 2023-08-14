close all; clear all;
N=100; % simulation time
T=0.1; % period

for i=1:4
    vector_matrix(:,:,i)=zeros(100,100);
end

c=0.2;
c=1e8;
%  do kk time experiments
for kk=1:100
    Q=eye(4);          % system process noise covariance matrix 
    R=0.1*eye(4);         % observation noise variance matrix, 4 observation states, horizontal x, vertical y,horizontal velocity,vertical velocity
    W=sqrt(Q)*randn(4,N); % system process noise
    V=sqrt(R)*randn(4,N);% observation noise
    A=[1,0,T,0;0,1,0,T;0,0,1,0;0,0,0,1]; % state transfer matrix

    H=eye(4);     % observation matrix
    % initialization
    X=zeros(4,N);  
    
    X(:,1)=[0;0;100;10]; % initial x position=0, y position=0. velocity x=100, velocity y=10;
    P0=diag([10,10,1,1]); % 

    Z=zeros(4,N);  
    Z(:,1)=[X(1,1);X(2,1);X(3,1);X(4,1)]; % init observation                                                                                 
    Xirls=zeros(4,N);% init irls estimation

    Xirls(:,1)=X(:,1)+sqrt(P0)*randn(4,1); % Introducing an estimated error variance during initialization to maintain alignment
    err_P=zeros(4,N); % Each column represents the four values of the Kalman's P matrix at that moment, with each row recording each timestamp.
    for k2=1:4
        err_P(k2,1)=P0(k2,k2);
    end
    I=eye(4);      % 4 dimensions systems

    % real error estimation
    for j=1:4
        vector=vector_matrix(:,:,j);
        vector(kk,1)=Xirls(j,1)-X(j,1);
        vector_matrix(:,:,j)=vector;
    end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for k=2:N
        % transfer equation
        X(:,k)=A*X(:,k-1)+W(:,k);
        
        % observation equation
        Z(:,k)=H*X(:,k)+V(:,k);

        % observation outlier
        if k==50
            Z(1:2,k)=1000*ones(2,1);
%             Z(k)=1000;
        end
    end
    
    for k=2:N

        % 1 state prediction
        Xirls_pre=A*Xirls(:,k-1);
        P_pre=A*P0*A'+Q;

        Lt=[R zeros(4,4);zeros(4,4) P_pre]; % Lt=Lk*Lk'
        % normalization
        Lk=Lt^(0.5);
        yk=Lk^(-1)*[Z(:,k);Xirls_pre];
        Ak=Lk^(-1)*[H;eye(4)];

        Kg=P_pre*H'*(H*P_pre*H'+R)^(-1); %

        % ri is a (3x1) vector
        ri=yk-Ak*Xirls_pre;
        
        Q1=zeros(8,8);
        % The derivative of the ρ function with respect to the vector ri.
        for k1=1:8
            ri_row=ri(k1); % the k1_th row value in ri
            if ri_row==0  % if ri_row=0,set the value of ri_row be a really small number
                ri_row=0.001;
            end
            
            roul_der=roul_derive(ri_row,c); % derivative
            Q1(k1,k1)=roul_der*(1/ri_row);  % Setting the values on the diagonal of matrix Q1.
        end        
        % Calculate the estimated value.
        Xirls_after=(Ak'*Q1*Ak)^(-1)*Ak'*Q1*yk;
        % Determine if the threshold is satisfied.
        while isConditioned(Xirls_after,Xirls_pre,0.01) 
            
            Xirls_pre=Xirls_after; % save
           
            ri=yk-Ak*Xirls_after;
            
            for k1=1:8
                ri_row=ri(k1);
                if ri_row==0
                    ri_row=0.001;
                end
            
                roul_der=roul_derive(ri_row,c);
                Q1(k1,k1)=roul_der*(1/ri_row);
            end
            %Q1_r(:,:,k)=Q1;
            Xirls_after=(Ak'*Q1*Ak)^(-1)*Ak'*Q1*yk;
        end
        
        P0=(I-Kg*H)*P_pre*(I-Kg*H)'+Kg*R*Kg'; % P_pre 应该写成 P(k-1)
        %P01=(Ak'*Ak)^(-1);
        Xirls(:,k)=Xirls_after; % Store the final iteration value of this moment into the Xirls matrix.

        % update error
        for j=1:4
            vector=vector_matrix(:,:,j);
            vector(kk,k)=Xirls(j,k)-X(j,k);
            vector_matrix(:,:,j)=vector;
        end

        for k2=1:4
            err_P(k2,k)=P0(k2,k2);
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

x_hat(:,1)=Xirls(:,1);
P(:,:,1)=diag([10,10,1,1]);
for k=2:N
    xp=A*x_hat(:,k-1);
    Pp=A*P(:,:,k-1)*A'+Q;
    K=Pp*H'*(H*Pp*H'+R)^(-1);
    x_hat(:,k)=xp+K*(Z(:,k)-H*xp);
    P(:,:,k)=(I-K*H)*Pp;
    
end

x_hat1(:,1)=Xirls(:,1);
P1(:,:,1)=diag([10,10,1,1]);
for k=2:N
    xp1=A*x_hat1(:,k-1);
    Pp1=A*P1(:,:,k-1)*A'+Q;
    
          
    Lt1=[R zeros(4,4);zeros(4,4) Pp1]; % Lt=Lk*Lk'
    % normalization
    Lk=Lt^(0.5);
    yk=Lk^(-1)*[Z(:,k);xp1];
    Ak=Lk^(-1)*[H;eye(4)];

    x_hat1(:,k)=(Ak'*Ak)^(-1)*Ak'*yk;
    P1(:,:,k)=(Ak'*Ak)^(-1);
    
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

% 
figure(1)
plot(Difference_matrix(:,:,1),'-bo'); % 
hold on;
plot(errPx_matrix(:,:,1),'-g+'); 
legend('irls real horizontal displacement error variance','kalman estimation variance');
xlabel('sampling time/s');
ylabel('variance');

figure(2)
plot(Difference_matrix(:,:,2),'-bo'); 
hold on;
plot(errPx_matrix(:,:,2),'-g+'); 
legend('IRLS real vertical error variance','kalman estimation error variance');
xlabel('sampling time/s');
ylabel('variance');

figure(3)
plot(Difference_matrix(:,:,3),'-bo'); 
hold on;
plot(errPx_matrix(:,:,3),'-g+');
legend('IRLS real horizontal velocity error variance','kalman estimation variance');
xlabel('sampling time/s');
ylabel('variance');

figure(4)
plot(Difference_matrix(:,:,4),'-bo');
hold on;
plot(errPx_matrix(:,:,4),'-g+');
legend('real IRLS vertical velocity error variance','kalman estimation error variance');
xlabel('sampling time/s');
ylabel('variance');

% horizontal displacement estimation
figure(5);
plot(X(1,:));
hold on;
plot(Xirls(1,:),'-.');
hold on;
plot(x_hat(1,:),'g--');
%plot(x_hat1(1,:),'y:');
legend('true state','IRLS estimate','KF estimate','LS estimate');

% Vertical Displacement Estimation
figure(6);
plot(X(2,:));
hold on;
plot(Xirls(2,:),'-.');
hold on;
plot(x_hat(2,:),'g--');
%plot(x_hat1(2,:),'y:');
legend('true state','IRLS estimate','KF estimate','LS estimate');

% horizontal velocity estimation
figure(7);
plot(X(3,:));
hold on;
plot(Xirls(3,:),'-.');
hold on;
plot(x_hat(3,:),'g--');
%plot(x_hat1(3,:),'y:');
legend('true state','IRLS estimate','KF estimate','LS estimate');

% Vertical velocity Estimation
figure(8);
plot(X(4,:));
hold on;
plot(Xirls(4,:),'-.');
hold on;
plot(x_hat(4,:),'g--');
%plot(x_hat1(4,:),'y:');
legend('true state','IRLS estimate','KF estimate','LS estimate');

figure(9)
plot(X(1,:),X(2,:),'*--'); % real position
hold on;
plot(Xirls(1,:),Xirls(2,:),'.--');
hold on;
plot(x_hat(1,:),x_hat(2,:),'r-');
legend('real position','irls estimation position','kalman estimation position');

% let l be the threshold, if all the value of (X_after(i)-X_pre(i)) less than l, then jump out of the loop.
% at least one of the (X_after(i)-X_pre(i)) bigger than l, then continue
% the loop.
function condition=isConditioned(X_pre,X_after,l)
    for i=1:4
        if X_after(i)-X_pre(i)<l
            continue
        else 
           condition=true;
           return;
        end
    end
    condition=false;
    return;
end
function roul_der = roul_derive(ri,c)
    if abs(ri)<c
        roul_der=ri;
    elseif ri>0
        roul_der=c;
    else
        roul_der=-c;
    end
end