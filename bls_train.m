function [A,A_test,W]=bls_train(X,Y,test_x,s,N1,Ng,N2)
lambda=1e-1;
Ns=N1*Ng;
X1 = [X .1 * ones(size(X,1),1)];
feature_nodes=zeros(size(X,1),Ng*N1);
for i=1:Ng
    Wr=2*rand(size(X,2)+1,N1)-1;
    A1 = X1 * Wr;
%     A1 = mapminmax(A1);% 归一化
    clear W;
    Ws  =  sparse_bls(A1,X1,1e-3,50)';% 产生稀疏矩阵
    We{i}=Ws;
    F1 = X1 * Ws;
%     fprintf(1,'Feature nodes in window %f: Max Val of Output %f Min Val %f\n',i,max(F1(:)),min(F1(:)));
    [F1,ps1]  =  mapminmax(F1',0,1);
    F1 = F1';
    ps(i)=ps1;
    feature_nodes(:,N1*(i-1)+1:N1*i)=F1;
end

clear X1;
clear F1;
%%%%%%%%%%%%%enhancement nodes%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X2 = [feature_nodes .1 * ones(size(feature_nodes,1),1)];
if Ns>=N2
    wh=orth(2*rand(Ns+1,N2)-1);
else
    wh=orth(2*rand(Ns+1,N2)'-1)'; 
end
enhancement_nodes = X2 *wh;
L2 = max(max(enhancement_nodes));
L2 = s/L2;
% fprintf(1,'Enhancement nodes: Max Val of Output %f Min Val %f\n',L2,min(enhancement_nodes(:)));
enhancement_nodes = tansig(enhancement_nodes * L2); %激活函数
A=[feature_nodes enhancement_nodes];
clear X2;clear enhancement_nodes;

XX1 = [test_x .1 * ones(size(test_x,1),1)];
feature_nodes_test=zeros(size(test_x,1),Ns);
for i=1:Ng
    Ws=We{i};ps1=ps(i);
    F2 = XX1 * Ws;
    F2  =  mapminmax('apply',F2',ps1)';
    clear Ws; clear ps1;
    feature_nodes_test(:,N1*(i-1)+1:N1*i)=F2;
end
clear F2;clear XX1;
XX2= [feature_nodes_test .1 * ones(size(feature_nodes_test,1),1)]; 
enhancement_nodes_test = tansig(XX2 * wh * L2);
A_test=[feature_nodes_test enhancement_nodes_test];
clear XX2;clear wh;clear enhancement_nodes_test;

W = (A'  *  A+eye(size(A',1)) * (lambda)) \ ( A'  *  Y);