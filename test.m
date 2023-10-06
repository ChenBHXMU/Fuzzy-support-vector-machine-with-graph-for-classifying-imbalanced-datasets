function [ACC1] = test()
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
myDataPath = {
    'WBC.mat'

    };
%初始化
[pathN,pathM]=size(myDataPath);
ACC1 = zeros(pathN,1);
TrainT1 = zeros(pathN,1);TestT1 = zeros(pathN,1);Svs1 = zeros(pathN,1);Std1 = zeros(pathN,1);
imbanlance = 1; %1 Gmean; 2 Fmean
typeonevone = 1;
%数据循环
for pathi = 1:pathN
    pathi
    matNameList = strsplit(char(myDataPath(pathi)),'\');
    [matNameN,matNameM] = size(matNameList);
    matName = matNameList(matNameM);
    matNameList = strsplit(char(matName),'.');
    matName = matNameList(1);
    %     if pathi < 28
    X = cell2mat(struct2cell(load(char(myDataPath(pathi)))));
    [nX,mX] = size(X);
    Y = X(:,mX);
    X = X(:,1:(mX-1));


    item = 1;


    for jj = 1:item

        X = mapminmax(X',0,1)';
        %item次五折交叉
        acca = zeros(5,1);trainTime1 = zeros(5,1);testTime1 = zeros(5,1);svS1 = zeros(5,1);
        data = [X,Y];
        [data_r, data_c] = size(data);
        indices = crossvalind('Kfold', data_r, 5);%5折交叉
        for i = 1 : 5
            test = (indices == i);
            train = ~test;
            test_data = data(test, 1 : data_c - 1);
            test_label = data(test, data_c);
            X=[test_data,test_label];
            train_data = data(train, 1 : data_c - 1);
            train_label = data(train, data_c);
            %adding label noise
            [train_label] = setLabelNoise(train_label,0.2);
            %parameter C
            paraList = [0.01,0.1,1,10,100];
            acc1Max = zeros(length(paraList),1);TrainTime1 = zeros(length(paraList),1);TestTime1 = zeros(length(paraList),1); sVs1 = zeros(length(paraList),1);

            %parfor iPara = 1:length(paraList)
            for iPara = 1:length(paraList)
                kertype = 'linear'; % linear linear-kernel  rbf rbf-kernel
                C = paraList(iPara);
                imbanlance = 1;
                [Acc1] = trainGraphSVM(train_data',train_label',test_data',test_label',kertype,C,imbanlance,0.5,typeonevone);
                acc1Max(iPara) = Acc1;
            end
            acca(i,1) = max(acc1Max);

        end
        ACC1(pathi,1) = mean(acca)
    end


end


end