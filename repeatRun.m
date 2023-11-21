%%所有数据集运行
name=["YaleB","LFW","AR"];
num=[10,15,20,25,5,6,7,8,4,6,8,12];
for i=2:length(name)
    num1=num(4*i-3:4*i);
    for j=1:length(num1)
        for k=1:40
            run_GReBLS(name(i),num1(j));
        end
    end
end

%%单个数据集调参
for i=1:10
    N1=randi([20,30],1,1);%feature nodes  per window
    Ng=randi([15,20],1,1);
    N2=randi([1000,2000],1,1);% number of enhancement nodes
    GReBLS1(name(i),num1(j));
end
