function [W] = revised_CLSR(X,Y,train_gt,opts)
%%常数项b含在了W中
[n,m]=size(X);
lambda=opts.lambda;
gamma=opts.gamma;
maxIter=opts.maxIter;
class_num=opts.nnClass;
%% cumsum 累计加和
each_train_number=zeros(1,class_num);
for i=1:class_num
each_train_number(i)=length(find(train_gt==i));
end
cum_each_train_number=cumsum(each_train_number);
cum_trainnumber = [0,cum_each_train_number];
%% 训练样本集加入一个常数列
X=[X 0.1 * ones(size(X,1),1)];
Q=(X'*X+lambda*eye(m+1))\X';
%% 最大迭代次数

A=zeros(n,class_num);

U=zeros(class_num);

ooxx=1:1:class_num;
iter=1;
while iter<=maxIter
    W=Q*(Y+A);
    %% 这部分代码没用到
%     for ii=1:class_num
%         class_index = cum_trainnumber(ii)+1:cum_trainnumber(ii+1);
% %         he(ii)=norm(A(class_index,:)-ones(length(class_index),1)*U(ii,:),'fro')^2;
%     end
%     obj(iter)=norm(X*W+en*b'-Y-A,'fro')^2+lambda*sum(he);
     %%
    P=X*W-Y;
    for i=1:n
       %% supt 是论文中的A_i,*; supr是G_i,*; supt(j)是Aij;supr(j)是Gij
        supt=zeros(1,class_num);
        supr=(P(i,:)+gamma*U(train_gt(i),:))/(1+gamma);
        k=train_gt(i);
        v=supr-supr(k);
        %% 返回ooxx不在k中的数组成的向量
        cc=setdiff(ooxx, k);
        triangle=0;
        %% 小写字母l赋为0
         l=0;
        for p=1:length(cc)
            %% v(cc(p))即为论文中v_p
            g=v(cc(p));
            for q=1:length(cc)
                g=g+min((v(cc(p))-v(cc(q))),0);
            end
            if g>0
                triangle=triangle+v(cc(p));
                l=l+1;
            end
        end
        
        triangle=triangle/(l+1);
        
        for j=1:class_num
            if j==k
                supt(j)=supr(j)+triangle;
            else
                supt(j)=supr(j)+min((triangle-v(j)),0);
            end
        end
        A(i,:)=supt;
    end
    for ii=1:class_num
        class_index = cum_trainnumber(ii)+1:cum_trainnumber(ii+1);
        %% U(ii,:) 求A的每类的行平均
        U(ii,:)=mean(A(class_index,:),1);
    end
    iter=iter+1;
end
%         plot(obj)
end