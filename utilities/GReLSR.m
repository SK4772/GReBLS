function [W,b] = GReLSR(X,Y,train_gt,opts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%X:ѵ��������
%Y:ѵ��������ǩ��
%train_gt:ѵ��������������
%opts(�ṹ��)��maxIter(����������)��nnClass(�������)��lambda��gamma������ϵ����
%%������b������W��
[n,m]=size(X);%n��ʾ��������m��ʾά��
%[~,c]=size(Y);
lambda=opts.lambda;%����ϵ��
gamma=opts.gamma;%����ϵ��
maxIter=opts.maxIter;%����������
class_num=opts.nnClass;%�������
%%cumsum �ۼƼӺ�
each_train_number=zeros(1,class_num);%���ڴ洢ÿ������������
%��ѭ�����ڻ�ȡÿ�����������������д洢
for i=1:class_num
each_train_number(i)=length(find(train_gt==i));
end
%�����ۻ���
cum_each_train_number=cumsum(each_train_number);
cum_trainnumber = [0,cum_each_train_number];%%0+�ۻ��͹���һ���µ�����
en=ones(n,1);%�����Ż�ʱ��en
ec=ones(class_num,1);%�����Ż�ʱ��ec

%%%%%%%%%%%%����Ϊ�Ż�%%%%%%%%%%%%%%%%%%%%%%
H=eye(n)-(1/n)*(en)*(en)';
Q=(X'*H*X+lambda*eye(m))\X'*H;%W=Q*T
%%����������
U=zeros(n,class_num);

a=zeros(n,1);

mu=zeros(class_num,1);

ooxx=1:1:class_num;%��1Ϊ��ʼ�㣬class_numΪ�յ㣬1Ϊ����������һ������(��㣺1���յ�)
iter=1;

rate=zeros(maxIter,1);
%%%%%%%%%%%%%%Optimnization GReLSR%%%%%%%%%%%%%%
while iter<=maxIter
    %%%%%%%step1%%%%%%%%%
    W= Q * (Y + Y .* U + a * ec');
    b=(Y + Y .* U + a * ec'-X*W)'*(en)/n;
    %%%%%%%step2%%%%%%%%% 
    R=X*W+en*b'-Y;
    %�����Ϊn��������
    for i=1:n  %%ÿһ�����
        yi=train_gt(i);%��ȡ��������
        for j=1:class_num
            if j==yi
                
                alpha =R(i,yi)+gamma;%%%�������²�:���ｫuk��ʼֵ��Ϊ��ȱ��uk��
                beta = 1+gamma; 
            end
        end
        R1(i,:)=sort(R(i,:),'descend');%��R���н�������
        for j=1:class_num
            if j~=yi
                R_ij=R1(i,j);
            end
        end
        s=0;
        for m=1:class_num
            if m~= yi
                for j=1:m-1
                    s=s+R_ij;
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
                U(i,j)=a(i,1)-R_ij;
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