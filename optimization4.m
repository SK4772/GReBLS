function [W] = optimization4(TrainX,TestX,TrainY,lambda1,lambda2,lambda3)
%OPTIMIZATION BLS+MMD+Double Weight+Label dragging
%此处显示详细说明

A1=TrainX;
A2=TestX;
Ys=TrainY;
A=[A1' A2']';
gnd=size(TrainY,2);
m=size(TrainX,2);
%%%初始化动态图以及权重矩阵W
T=eye(gnd);

s=size(TrainX,1);
t=size(TestX,1);
M=zeros(s,gnd);
B=2*TrainY-1;
Vm=Construct_Vm(s,t);
W1=A1'*A1+lambda1*A'*Vm*A+lambda2*eye(size(A1,2)); W2=A1'*TrainY;
W=W1\W2; P=W;

for iters=1:30
    %%%optimization:W
    W=(lambda2*eye(m)+A'*Vm*A)\(lambda2*P*T);

    %%%optimization:P
    P1=lambda1*A1'*A1;
    P2=lambda2*T*T';
    P3=lambda1*A1'*(Ys+B.*M)+lambda2*W*T';
    P=sylvester(P1,P2,P3);
    %%%optimization:T
    T=(P'*P+lambda3*eye(gnd))\(P'*W);
    %%%optimization:M
    M=max(B.*(A1*P-Ys),0);

end




