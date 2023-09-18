%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function main3_3_2
% non-linear model， No OLS, No IRLS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
N=100; % simulation time
T=0.1;

kx=0.01;ky=0.05;g=9.8;

n=4;
weight=1/(2*n+1);

    Q=diag([0.5,1,0.5,1]);          % system process noise covariance matrix 
    R=diag([1,1]);         % observation noise variance matrix,  observation states, distance, angle   
    % system noise and observation noise
    W=sqrt(Q)*randn(4,N);
    V=sqrt(R)*randn(2,N);

    % initialization
    X=zeros(4,N);  
    Z=zeros(2,N);  

    X(:,1)=[0,50,500,0]; % initial x position=0,velocity x,y position, velocity y;
    P0=eye(4);  
    
    % init system state
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

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Estimation 1 
    % current time pre-estimation UKF
    X_ukf=zeros(4,N);% init estimation

    X_ukf(:,1)=X(:,1)+sqrt(P0)*randn(4,1); % Introducing an estimated error variance during initialization to maintain alignment    err_P=zeros(4,N); % Each column represents the four values of the Kalman's P matrix at that moment, with each row recording each timestamp.

    I=eye(4);      % 4 dimensions systems

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for k=2:N
        % acquire a set of sigma samplings point
        % the first sampling point
        Xestimate=X_ukf(:,k-1);
        % 1~2n point
        P_chol=(n*P0)^(1/2);

        for j=1:n
            X_sigmap1(:,j)=Xestimate+P_chol(:,j);
            X_sigmap2(:,j)=Xestimate-P_chol(:,j);
        end

        % 2n+1 sigma points
        X_sigma=[Xestimate X_sigmap1 X_sigmap2];
        
        % step 1: computing  prediction of sigma points  of time k
        X_sigmapre=zeros(n,2*n+1);
 
        % element-wise, using .^
        x1=X_sigma(1,:)+X_sigma(2,:)*T;
        v1=X_sigma(2,:)-kx*X_sigma(2,:).^(2)*T;
        y1=X_sigma(3,:)-X_sigma(4,:)*T;
        v2=X_sigma(4,:)-(ky*X_sigma(4,:).^(2)-g)*T;
        X_sigmapre=[x1;v1;y1;v2];
        % state prediction, apply the state prediction to the variable Xukf)

        % step 2: computing the prediction of normal points of time k
        X_pre=zeros(4,1);
        for j=1:2*n+1
            X_pre=X_pre+weight*X_sigmapre(:,j);
        end
        P_pre=zeros(4,4);
        for j=1:2*n+1
            P_pre=P_pre+weight*(X_sigmapre(:,j)-X_pre)*(X_sigmapre(:,j)-X_pre)';
        end
        P_pre=P_pre+Q;

        % step 3 : computing new P_chol
        P_chol=(n*P_pre)^(1/2);
        % 1~2n points
        for j=1:n
            X_sigmap1(:,j)=X_pre+P_chol(:,j);
            X_sigmap2(:,j)=X_pre-P_chol(:,j);
        end
        
        % 2n+1 point
        X_sigma1=[X_pre X_sigmap1 X_sigmap2];
        
        % computing Z_pre
        % step : the prediction of Z_sigmapre
        %Z_sigmapre=zeros(2,2*n+1);
        x1=X_sigma1(1,:);y1=X_sigma1(3,:);
        r=(x1.*x1+y1.*y1).^(1/2);
        alpha1=atan(x1./y1)*180/pi;
        Z_sigmapre=[r;alpha1];
        
        % step: computing the prediction of Z_pre
        Z_pre=zeros(2,1);
        for j=1:2*n+1
            Z_pre=Z_pre+weight*Z_sigmapre(:,j);
        end
        % linearization

        % step 3.1:
        % computing Pxgama
        Pxgama=zeros(n,n);
        for j=1:2*n+1
            Pxgama=Pxgama+weight*(X_sigma(:,j)-Xestimate)*(X_sigma1(:,j)-X_pre)';
        end
        % computing Pxx(k-1|k-1) and Fk, Fk be the parameter of statistical
        % linearization
        Pxxlast=zeros(n,n);
        for j=1:2*n+1
            Pxxlast=Pxxlast+weight*(X_sigma(:,j)-Xestimate)*(X_sigma(:,j)-Xestimate)';
        end
        Fk=Pxgama'*P_pre^(-1); % using P_pre or Pxxlast is ok.
        
        % computing Pxz
        Pxz=zeros(4,2);
        for j=1:2*n+1
            Pxz=Pxz+weight*(X_sigma1(:,j)-X_pre)*(Z_sigmapre(:,j)-Z_pre)';
        end

        % computing Hk: Hk be the parameter of statistical linearization
        Hk=Pxz'*P_pre^(-1);

        % step 6: computing Pzz
        Pzz=zeros(2,2);
        for j=1:2*n+1
            Pzz=Pzz+weight*(Z_sigmapre(:,j)-Z_pre)*(Z_sigmapre(:,j)-Z_pre)';
        end
        
        Pzz=Pzz+R; % adding R to Pzz or not is okay.
        
        % computing paiK: paiK be the error on the nonliear measurement
        % function's covariance matrix
        paiK=Pzz-Pxz'*P_pre^(-1)*Pxz-R;

        % Y=Hx+e linearization
        Y_tilta=[Z(:,k)+Hk*X_pre-Z_pre;X_pre];
        H_tilta=[Hk;eye(4)]; % H_tilta be 2x4
        R_tilta=[R+paiK zeros(2,4);zeros(4,2) P_pre];

        X_est=(H_tilta'*R_tilta^(-1)*H_tilta)^(-1)*H_tilta'*R_tilta^(-1)*Y_tilta;

        % step 9: computing state update and covariance update
        X_ukf(:,k)=X_est;
        P0=(H_tilta'*R_tilta^(-1)*H_tilta)^(-1);
        
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% not use regression UKF

% Q,R,X,Z,W,err_P has already been initialized during the regression
P0=eye(4);
X_ukfnoregression=zeros(4,N);
X_ukfnoregression(:,1)=X_ukf(:,1);
%X(:,1)+sqrt(P0)*randn(4,1);

for k=2:N
    Xestimate=X_ukfnoregression(:,k-1);
    % 1~2n point
    P_chol=(n*P0)^(1/2);

    for j=1:n
        X_sigmap1(:,j)=Xestimate+P_chol(:,j);
        X_sigmap2(:,j)=Xestimate-P_chol(:,j);
    end
    X_sigma=[Xestimate X_sigmap1 X_sigmap2];
    X_sigmapre=zeros(n,2*n+1);

    x1=X_sigma(1,:)+X_sigma(2,:)*T;
    v1=X_sigma(2,:)-kx*X_sigma(2,:).^(2)*T;
    y1=X_sigma(3,:)-X_sigma(4,:)*T;
    v2=X_sigma(4,:)-(ky*X_sigma(4,:).^(2)-g)*T;
    X_sigmapre=[x1;v1;y1;v2];

    X_pre=zeros(4,1);
    for j=1:2*n+1
        X_pre=X_pre+weight*X_sigmapre(:,j);
    end
    P_pre=zeros(4,4);
    for j=1:2*n+1
        P_pre=P_pre+weight*(X_sigmapre(:,j)-X_pre)*(X_sigmapre(:,j)-X_pre)';
    end
    P_pre=P_pre+Q;

    P_chol=(n*P_pre)^(1/2);
    for j=1:n
        X_sigmap1(:,j)=X_pre+P_chol(:,j);
        X_sigmap2(:,j)=X_pre-P_chol(:,j);
    end

    X_sigma=[X_pre X_sigmap1 X_sigmap2];

    x1=X_sigma(1,:);y1=X_sigma(3,:);r=(x1.*x1+y1.*y1).^(1/2);
    alpha1=atan(x1./y1)*180/pi;Z_sigmapre=[r;alpha1];

    Z_pre=zeros(2,1);
    for j=1:2*n+1
        Z_pre=Z_pre+weight*Z_sigmapre(:,j);
    end
    Pzz=zeros(2,2);
    for j=1:2*n+1
        Pzz=Pzz+weight*(Z_sigmapre(:,j)-Z_pre)*(Z_sigmapre(:,j)-Z_pre)';
    end
    Pzz=Pzz+R;

    Pxz=zeros(4,2);
    for j=1:2*n+1
        Pxz=Pxz+weight*(X_sigma(:,j)-X_pre)*(Z_sigmapre(:,j)-Z_pre)';
    end

    K=Pxz*Pzz^(-1);
    X_ukfnoregression(:,k)=X_pre+K*(Z(:,k)-Z_pre);
    P0=P_pre-K*Pzz*K';

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% time k-1 optimal estimation UKF
P0=eye(4);
X_regression=zeros(4,N);
X_regression(:,1)=X_ukf(:,1);

for k=2:N
        Xestimate=X_regression(:,k-1);
        % 1~2n point
        P_chol=(n*P0)^(1/2);

        for j=1:n
            X_sigmap1(:,j)=Xestimate+P_chol(:,j);
            X_sigmap2(:,j)=Xestimate-P_chol(:,j);
        end

        % 2n+1 sigma points
        X_sigma=[Xestimate X_sigmap1 X_sigmap2];
        
        % step 1: computing  prediction of sigma points  of time k
        X_sigmapre=zeros(n,2*n+1);
 
        % element-wise, using .^
        x1=X_sigma(1,:)+X_sigma(2,:)*T;
        v1=X_sigma(2,:)-kx*X_sigma(2,:).^(2)*T;
        y1=X_sigma(3,:)-X_sigma(4,:)*T;
        v2=X_sigma(4,:)-(ky*X_sigma(4,:).^(2)-g)*T;
        X_sigmapre=[x1;v1;y1;v2];

        % step 2: computing the prediction of normal points of time k
        X_pre=zeros(4,1);
        for j=1:2*n+1
            X_pre=X_pre+weight*X_sigmapre(:,j);
        end
        P_pre=zeros(4,4);
        for j=1:2*n+1
            P_pre=P_pre+weight*(X_sigmapre(:,j)-X_pre)*(X_sigmapre(:,j)-X_pre)';
        end
        P_pre=P_pre+Q;

        % step 3 : computing new P_chol
        P_chol=(n*P_pre)^(1/2);
        % 1~2n points
        for j=1:n
            X_sigmap1(:,j)=X_pre+P_chol(:,j);
            X_sigmap2(:,j)=X_pre-P_chol(:,j);
        end
        
        % 2n+1 point
        X_sigma1=[X_pre X_sigmap1 X_sigmap2];
        
        % computing Z_pre
        % step : the prediction of Z_sigmapre
        %Z_sigmapre=zeros(2,2*n+1);
        x1=X_sigma1(1,:);y1=X_sigma1(3,:);
        r=(x1.*x1+y1.*y1).^(1/2);
        alpha1=atan(x1./y1)*180/pi;
        Z_sigmapre=[r;alpha1];
        
        % step: computing the prediction of Z_pre
        Z_pre=zeros(2,1);
        for j=1:2*n+1
            Z_pre=Z_pre+weight*Z_sigmapre(:,j);
        end
        % linearization

        % step 3.1:
        % computing Pxgama
        Pxgama=zeros(n,n);
        for j=1:2*n+1
            Pxgama=Pxgama+weight*(X_sigma(:,j)-Xestimate)*(X_sigma1(:,j)-X_pre)';
        end
        % computing Pxx(k-1|k-1) and Fk, Fk be the parameter of statistical
        % linearization
        Pxxlast=zeros(n,n);
        for j=1:2*n+1
            Pxxlast=Pxxlast+weight*(X_sigma(:,j)-Xestimate)*(X_sigma(:,j)-Xestimate)';
        end
        Fk=Pxgama'*Pxxlast^(-1); % using Pxxlast
        

        % computing Pxz
        Pxz=zeros(4,2);
        for j=1:2*n+1
            Pxz=Pxz+weight*(X_sigma1(:,j)-X_pre)*(Z_sigmapre(:,j)-Z_pre)';
        end
        
        % computing Pxzpre
        Pxzpre=zeros(4,2);
        for j=1:2*n+1
            Pxzpre=Pxzpre+weight*(X_sigma(:,j)-Xestimate)*(Z_sigmapre(:,j)-Z_pre)';
        end


        % computing Hk: Hk be the parameter of statistical linearization
        Hk=Pxz'*P_pre^(-1);

        % step 6: computing Pzz
        Pzz=zeros(2,2);
        for j=1:2*n+1
            Pzz=Pzz+weight*(Z_sigmapre(:,j)-Z_pre)*(Z_sigmapre(:,j)-Z_pre)';
        end
        
        Pzz=Pzz+R; % adding R to Pzz 
        % computing paiK: paiK be the error on the nonliear measurement
        % function's covariance matrix
        paiK=Pzz-Pxz'*P_pre^(-1)*Pxz-R;  
        % computing Lk
        Lk=P_pre-Pxgama'*Pxxlast^(-1)*Pxgama-Q;
        
        % computing Pgamaz
        P_gamaz=zeros(4,2);
        for j=1:2*n+1
            P_gamaz=P_gamaz+weight*(X_sigma(:,j)-Xestimate)*(Z_sigmapre(:,j)-Z_pre)';
        end
        % computing Peksai: ek and kesaik be the statistical linearization
        % error covariance
        %P_eksai=zeros(4,2);

        % A1:Fk  A2:Hk  
        P_eksai=Pxz-(P_pre-Q)*Hk'-Fk*P_gamaz+Fk*Pxgama*Hk';
        P_eksai=-P_eksai;

        % computing P_deltae
        P_deltae=Pxgama-Pxxlast*Fk';
        % computing P_deltaksai
        P_deltaksai=-Pxzpre+Pxgama*Hk';

        %linearization
        Y_tilta=[Xestimate;X_pre-Fk*Xestimate;Z(:,k)-Z_pre+Hk*X_pre];
        H_tilta=[I zeros(4,4);-Fk I;zeros(2,4) Hk];
        R_tilta=[P_pre P_deltae P_deltaksai;P_deltae' Q+Lk P_eksai;P_deltaksai' P_eksai' paiK+R];

        X_tilta=(H_tilta'*R_tilta^(-1)*H_tilta)^(-1)*H_tilta'*R_tilta^(-1)*Y_tilta;
        P_tilta=(H_tilta'*R_tilta^(-1)*H_tilta)^(-1);
        
        X_regression(:,k)=X_tilta(5:8); % select row 5~8 of the augmented X_tilta
        P_pre=P_tilta(5:8,5:8); % select row 5~8 and column 5~8 of the augmented P_tilta

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% comparing the difference between classical KF and UKF
figure(1)
plot(X_ukf(1,:),'-go')
hold on;
plot(X_ukfnoregression(1,:),'b--');
plot(X(1,:),'-ro');
plot(X_regression(1,:),'r--');
legend('UKF regression1 X displacement estimation','UKFnoregression X displacement estimation','real X displacement estimation','UKF_regression2 X displacement estimation');
xlabel('sampling time/s');
ylabel('displacement/m');

figure(2)
plot(X_ukf(2,:),'-go')
hold on;
plot(X_ukfnoregression(2,:),'b--');
plot(X(2,:),'-ro');
plot(X_regression(2,:),'r--');
legend('UKF_regression1 X velocity estimation','UKFnoregression X velocity estimation','real X velocity estimation','UKF_regression2 X velocity estimation');
xlabel('sampling time/s');
ylabel('velocity m/s');

function d=Dist(X1,X2)
    d=sqrt(X1^(2)+X2^(2));
end
