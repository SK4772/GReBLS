function [W,b] = optimization_GReLSR(X,Y,train_gt,opts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%X:训练样本集
%Y:训练样本标签集
%train_gt:训练样本所属分类
%opts(结构体)：maxIter(最大迭代次数)，nnClass(类的数量)，lambda，gamma（正则化系数）
%%常数项b不含在W中
[n,m]=size(X);%n表示样本数，m表示维度
%[~,c]=size(Y);
lambda=opts.lambda;%正则化系数
gamma=opts.gamma;%正则化系数
maxIter=opts.maxIter;%最大迭代次数
class_num=opts.nnClass;%类的数量
%%cumsum 累计加和
each_train_number=zeros(1,class_num);%用于存储每类样本的数量
%该循环用于获取每类样本的数量并进行存储
for i=1:class_num
each_train_number(i)=length(find(train_gt==i));
end
%返回累积和
cum_each_train_number=cumsum(each_train_number);
cum_trainnumber = [0,cum_each_train_number];%%0+累积和构成一个新的数组
en=ones(n,1);%构建优化时的en
ec=ones(class_num,1);%构建优化时的ec

%%%%%%%%%%%%以下为优化%%%%%%%%%%%%%%%%%%%%%%
H=eye(n)-(1/n)*(en)*(en)';
lambda=lambda*trace(X'*H*X)/m;
Q=(X'*H*X+lambda*eye(m))\X'*H;%W=Q*T
%%最大迭代次数
U=zeros(n,class_num);

a=zeros(n,1);

mu=zeros(class_num,1);

ooxx=1:1:class_num;%以1为起始点，class_num为终点，1为步长，生成一个向量(起点：1：终点)
iter=1;

rate=zeros(maxIter,1);
%%%%%%%%%%%%%%Optimnization GReLSR%%%%%%%%%%%%%%
while iter<=maxIter
    %%%%%%%step1%%%%%%%%%
    W= Q * (Y + Y .* U + a * ec');
    b=(Y + Y .* U + a * ec'-X*W)'*(en)/n;
    %%%%%%%step2%%%%%%%%% 
    R=X*W+en*b'-Y;
    %拆分行为n个子问题
    for i=1:n  %%第i行
         yi=train_gt(i);%获取所属分类yi
        ttttt=gamma*mu(yi);
        alpha =R(i,yi)+gamma*mu(yi);
        beta = 1+gamma;
        R1(i,:)=sort(R(i,:),'descend');%对R进行降序排序
        % for j=1:class_num
        %     if j~=yi
        %         R_ij=R1(i,j);
        %     end
        % end
        
        for m=1:class_num
            s=0;
            if m~= yi
                for j=1:m-1
                    s=s+R1(i,j);
                end               
            end
            a(i,1)= (s+alpha)/(m-1+beta);
            if a(i,1)>=R1(i,m)
            break;
            end
        end
        for j=1:class_num
            if j <= m-1
                U(i,j)=0;
            else
                U(i,j)=a(i,1)-R1(i,j);
            end
        end
           
    end
    
   
   
     for ii=1:class_num
        class_index = cum_trainnumber(ii)+1:cum_trainnumber(ii+1);
        mu(ii,1)=mean(a(class_index));
    end
    iter=iter+1;
end

end