function [maxAcc,SVs,trainTime,testTime,weight,alpha] = trainGraphSVM(trainData,trainLabel,testData,testLabel,kertype,C,imbanlance,k,typeonevone)
%trainData dim*n trainLabel 1*n
% X = [trainData;testData];
% X = mapminmax(X',0,1)';

[mtrain,ntrain] = size(trainData);
tList = [0.05,0.1,0.5,1,5];
%tList = [1,2,5,8];
result = zeros(length(tList),1);
SvsList = zeros(length(tList),1);
trainTimeList = zeros(length(tList),1);
testTimeList = zeros(length(tList),1);
WeightList = zeros(ntrain,length(tList));
alphaList = zeros(ntrain,length(tList));
for i = 1:length(tList)
    [Acc,SVs,~,trainTime,testTime,~,svm,weight] = svmGraphSVMTrain_multiclass(trainData,trainLabel,testData,testLabel,kertype,C,tList(i),imbanlance,k,typeonevone);
    result(i,1) = Acc;
    SvsList(i,1) = SVs;
    trainTimeList(i,1) = trainTime;
    testTimeList(i,1) = testTime;
%    WeightList(:,i) = weight;
    %alphaList(:,i) = svm.aAll;
end
[maxAcc,index] = max(result);
SVs = SvsList(index);
trainTime = trainTimeList(index);
testTime = testTimeList(index);
weight = WeightList(:,index);
alpha = alphaList(:,index);
end

