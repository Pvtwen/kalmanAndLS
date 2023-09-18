%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function main3_3_2 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
N=100; %
    % 噪声
    Q=[1,0;0,1]; % 
    R=1;         % 
    W=sqrt(Q)*randn(2,N);% 
    V=sqrt(R)*randn(1,N);%
    % 系统矩阵
    A=[1,1;0,1]; %

    H=[1,0];     %
    % init
    X=zeros(2,N);  % 
    X(:,1)=[95;1]; % 
    P0=[10,0;0,1]; % 

    Z=zeros(1,N);  
    Z(1)=H*X(:,1); %                                                                                  
    Xls=zeros(2,N);% 
    Xls_rec=zeros(2,N);

    Xls(:,1)=X(:,1)+sqrt(P0)*randn(2,1); % LS estimation variable
    Xls_rec=Xls(:,1);
    
    I=eye(2);      % 
    for k=2:N
        X(:,k)=A*X(:,k-1)+W(k);
        Z(k)=H*X(:,k)+V(k);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for k=2:N
  
        P_pre=A*P0*A'+Q;
        % the parameter of linearization after augmented
        Y_tilde=[Xls(:,k-1);zeros(2,1);Z(k)];
        H_tilde=[I zeros(2,2);A -I;zeros(1,2) H];
        R_tilde=[P0 zeros(2,2) zeros(2,1);zeros(2,2) Q zeros(2,1);zeros(1,2) zeros(1,2) R];
        
        Xestimation=(H_tilde'*R_tilde^(-1)*H_tilde)^(-1)*H_tilde'*R_tilde^(-1)*Y_tilde; % (HRH)HRy
        Xls(:,k)=Xestimation(3:4); % row 1,2 is X_{k-1}, row 3,4 is X_{k}

        % Kalman gaim
        Kg=P_pre*H'*(R+H*P_pre*H')^(-1);
        
        P=(H_tilde'*R_tilde^(-1)*H_tilde)^(-1);% update covariance matrix
        %P0=(I-Kg*H)*P_pre;
        P0=P(3:4,3:4); % row 1,2 and column 1,2 is P_{k-1|k-1}
    end

    P0=[10,0;0,1]; % redo
    for k=2:N
        Xls_pre=A*Xls_rec(:,k-1);
        P_pre=Q+A*P0*A';

        K=P_pre*H'*(R+H*P_pre*H')^(-1);
        Xls_rec(:,k)=Xls_pre+K*(Z(:,k)-H*Xls_pre);
        P0=(I-K*H)*P_pre;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%end
figure
plot(Xls(1,:),'bo');
hold on;
plot(X(1,:),'-ro');
plot(Xls_rec(1,:),'b--');
legend('kalman estimation','real','kalman rec estimation');
xlabel('sampling time/s');
ylabel('value');