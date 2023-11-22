clc
clear
close all

%parpool(4)
%code of A GReLSR model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%数据加载%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('data'));%添加数据集所在路径
addpath(genpath('utilities'));%添加函数所在路径
name = 'AR';%加载数据集名称
load (name);%加载数据集

%%%%%%%%%%%%%%%%%%%%%%%%%%%%参数设置%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N1=randi([15,30],1,1);%feature nodes  per window
Ng=randi([15,30],1,1);
N2=randi([1000,3000],1,1);% number of enhancement nodes
s = .8;
%%两个参数的变化范围
lambdaE=[1e+4 1e+3 1e+2 1e+1  1 1e-1 1e-2 1e-3 1e-4];
gammaF=[1e+4 1e+3 1e+2 1e+1  1 1e-1 1e-2 1e-3 ];
%%每类筛选出训练样本的个数
select_num = 8;
%%最大迭代次数
maxIter=100;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%数据变量%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Train_X = [];
Train_Y = [];
Test_X = [];
Test_Y = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%数据处理%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[~,gnd] = max(Y,[],2);
fea = X;
fea = double(fea);
%%nnClass 类别总数
nnClass = length(unique(gnd));  % The number of classes;
num_Class = [];%每类中所含样本数目
for i = 1:nnClass
    num_Class = [num_Class length(find(gnd==i))]; %The number of samples of each class
end
%%%%数据分离，分为测试数据和训练数据%%%%
for j = 1:nnClass
    idx      = find(gnd==j);%select samples id per class
    randIdx  = randperm(num_Class(j));%每类样本中的样本数为num_Class(j)，将他们随机排列
    %从中随机取得select_num个样本当作训练数据
    Train_X = [Train_X; fea(idx(randIdx(1:select_num)),:)];            % select select_num samples per class for training
    Train_Y= [Train_Y;gnd(idx(randIdx(1:select_num)))];
    %其余样本当作测试数据
    Test_X  = [Test_X;fea(idx(randIdx(select_num+1:num_Class(j))),:)];  % select remaining samples per class for test
    Test_Y = [Test_Y;gnd(idx(randIdx(select_num+1:num_Class(j))))];
end
%%%%数据归一化处理%%%%
Train_X = Train_X';                       % transform to a sample per column
Train_X = Train_X./repmat(sqrt(sum(Train_X.^2)),[size(Train_X,1) 1]);
Test_X  = Test_X';
Test_X  = Test_X./repmat(sqrt(sum(Test_X.^2)),[size(Test_X,1) 1]);
label = unique(Train_Y);
Y = bsxfun(@eq, Train_Y, label');
Y = double(Y);
X = Train_X';
[X,X1,~]=bls_train(X,Train_Y,Test_X',s,N1,Ng,N2);
Test_X=X1';
%%%%定义用于存储结果的数据%%%%
accuracy=0;%准确率（最大值）
lambda=0;
gamma=0;
rate_acc = zeros(length(lambdaE),length(gammaF));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%数据训练（无偏置）%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:length(lambdaE)
    lambda = lambdaE(i);
    parfor j=1:length(gammaF)
        fprintf("%d:The lambda is %e and The gammaF is %e \n",i,lambdaE(i),gammaF(j));
        gamma = gammaF(j);
        [W] = optimization_GReBLS1(X,Y,Train_Y',maxIter,gamma,lambda,nnClass);
        predict_label=zeros(size(Test_X,2),1);
        for ii=1:size(Test_X,2)
            %%y是测试样本； test是测试样本集
            y=Test_X(:,ii);
            reconstruction=W'*y;
            [~,predict_label(ii)]=max(reconstruction);
        end
        rate_acc(i,j) = calcError(Test_Y, predict_label, 1:nnClass);

    end
    acc=max(max(rate_acc));
    [M,N] = find (rate_acc == max(max(rate_acc)));
    accuracy=acc(1);
    lambda=lambdaE(M(1));
    gamma=gammaF(N(1));
end
save ( ['result\0Result_' name '_' num2str(select_num) '_' num2str(accuracy) '_' num2str(N1) '_' num2str(N2) '_' num2str(Ng) '.mat'], 'accuracy', 'lambda', 'gamma', 'rate_acc','N1','N2','Ng');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%数据训练%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
accuracy=0;%准确率（最大值）
lambda=0;
gamma=0;
rate_acc = zeros(length(lambdaE),length(gammaF));
for i = 1:length(lambdaE)
    lambda = lambdaE(i);
    parfor j=1:length(gammaF)
        fprintf("%d:The lambda is %e and The gammaF is %e \n",i,lambdaE(i),gammaF(j));
        gamma = gammaF(j);
        [W,b] = optimization_GReBLS(X,Y,Train_Y',maxIter,gamma,lambda,nnClass);
        predict_label=zeros(size(Test_X,2),1);
        for ii=1:size(Test_X,2)
            %%y是测试样本； test是测试样本集
            y=Test_X(:,ii);
            reconstruction=W'*y+b;
            [~,predict_label(ii)]=max(reconstruction);
        end
        rate_acc(i,j) = calcError(Test_Y, predict_label, 1:nnClass);

    end
    acc=max(max(rate_acc));
    [M,N] = find (rate_acc == max(max(rate_acc)));
    accuracy=acc(1);
    lambda=lambdaE(M(1));
    gamma=gammaF(N(1));
end
save ( ['result\Result_' name '_' num2str(select_num) '_' num2str(accuracy) '_' num2str(N1) '_' num2str(N2) '_' num2str(Ng) '.mat'], 'accuracy', 'lambda', 'gamma', 'rate_acc','N1','N2','Ng');

