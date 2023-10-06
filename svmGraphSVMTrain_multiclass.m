function [Acc,SVs,preY,trainTime,testTime,proList,svm,Fweight] = svmGraphSVMTrain_multiclass( trainData,trainLabel,testData,testLabel,kertype,C,t,imbanlance,k,type)
%trainData dim*n  trainLabel 1*n

if isempty(k)
    k = 1;
end
SVs = 0;%支持向量个数
class = unique(trainLabel);
nuclass = length(class);
trainTime = 0;
testTime = 0;
[mTest,nTest] = size(testData);
testYList = zeros(nuclass,nTest);
proList = zeros(nuclass,nTest);
[mTrain,nTrain] = size(trainData);
Fweight = ones(nTrain,1);
epsilon=1e-5;
if(nuclass == 2)
    preY = zeros(1,nTest);
    trainLabel = mapminmax(trainLabel,-1,1);
    index1 = find(trainLabel == 1); index2 = find(trainLabel == -1);
    trainData1 = trainData(:,index1);trainLabel1 = trainLabel(:,index1);
    trainData2 = trainData(:,index2);trainLabel2 = trainLabel(:,index2);
    trainData = [trainData1,trainData2];trainLabel = [trainLabel1,trainLabel2];
    tic;
    Fweight = getGraphSVMFuzzyWeight(trainData1,trainLabel1,trainData2,trainLabel2,t,k);
    %trainData = [trainData(:,index1),trainData(:,index2)];trainLabel = [trainLabel(:,index1),trainLabel(:,index2)];
    options=optimset;
    options.LargerScale='off';
    options.Display='off';
    n=length(trainLabel);
    H=(trainLabel'*trainLabel).*kernel(trainData,trainData,kertype);
    f=-ones(n,1);
    A=[];
    b=[];
    Aeq=trainLabel;
    beq=0;
    lb=zeros(n,1);
    ub=C*Fweight;
    a0=zeros(n,1);
    [a,fval,eXitflag,output,lambda]=quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
    [svm,sv_label] = calculate_rho(a,trainData',trainLabel',C,kertype);
    trainTime = toc;
    SVs = SVs + svm.svnum;
    
    tic;
    result=svmTest_multiclass(svm,testData,kertype);
    testTime = toc;
    indPreYmin = find(result.score<0);
    preY(1,indPreYmin) = min(class);
    indPreYmax = find(result.score>0);
    preY(1,indPreYmax) = max(class);
    %得到精度 
    if imbanlance ==1
        Acc = Gmean(preY,testLabel);
    elseif imbanlance == 2
        Acc = Fmean(preY,testLabel);
    elseif imbanlance == 0
        Acc = size(find(preY==testLabel))/size(testLabel);
    elseif imbanlance == 3
        %get probability
        for ipro = 1:length(testLabel)
            if(preY(ipro) == -1)
                proList(1,ipro) = getProbability(result.score(ipro));
                proList(2,ipro) = 1- proList(1,ipro);
            else
                proList(2,ipro) = getProbability(result.score(ipro));
                proList(1,ipro) = 1- proList(2,ipro);
            end
        end
        [~,~,Acc] = calculate_roc(proList(1,:), testLabel);
    end
elseif(nuclass > 2 && type == 1) %max(f(x))
    testY  = []; %保存最后的标签
    %     struct SVM;
    nn = 0;
    [mTrain,nTrain] = size(trainData);
    svsList = zeros(nTrain,1); %用来记录支持向量的个数，非0即支持向量
    for ii = 0:nuclass-1
        nn = nn + 1;
        iclass = class(ii+1);
        iindex = find(trainLabel==iclass);
        %one v all
        jindex = find(trainLabel~=iclass);
        itrainLabel = trainLabel(iindex) - trainLabel(iindex) + 1;%one为1
        jtrainLabel = trainLabel(jindex) - trainLabel(jindex) - 1;%All为-1
        itrainData = trainData(:,iindex);
        jtrainData = trainData(:,jindex);
        ijtrainLabel = [itrainLabel,jtrainLabel];
        ijtrainData = [itrainData,jtrainData];
        Fweight = getGraphSVMFuzzyWeight(itrainData,itrainLabel,jtrainData,jtrainLabel,t,k);
        options=optimset;
        options.LargerScale='off';
        options.Display='off';
        
        n=length(ijtrainLabel);
        H=(ijtrainLabel'*ijtrainLabel).*kernel(ijtrainData,ijtrainData,kertype);
        f=-ones(n,1);
        A=[];
        b=[];
        Aeq=ijtrainLabel;
        beq=0;
        lb=zeros(n,1);
        ub=C*Fweight;
        a0=zeros(n,1);
        tic;
        [a,fval,eXitflag,output,lambda]=quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
        %求b
        [svm,sv_label] = calculate_rho(a,ijtrainData',ijtrainLabel',C,kertype);
        svsList(sv_label(:,:))=1;%单次支持向量
        
        svm.a=a(sv_label);
        svm.Xsv=ijtrainData(:,sv_label);
        svm.Ysv=ijtrainLabel(sv_label);
        %SVs = SVs + svm.svnum;
        trainTime = trainTime + toc;
        tic;
        result=svmTest_multiclass(svm,testData,kertype);
        testTime = testTime + toc;
        testYList(nn,:) = result.score;
    end
    [maxLabel,maxIndex] = max(testYList);
    testY = class(maxIndex);
    %得到精度
    preY = testY;
    %Acc = size(find(testLabel==preY))/size(testLabel);
    if imbanlance ~=2
        Acc = Gmean(preY,testLabel);
    elseif imbanlance == 2
        Acc = Fmean(preY,testLabel);
    end
    svm.svnum = length(find(svsList==1));
    SVs = SVs + svm.svnum;
    proList = softmax(testYList);
elseif(nuclass > 2 && type == 2) %1v1
    testY  = []; %保存最后的标签
    %     struct SVM;
    nn = 0;
    [mTrain,nTrain] = size(trainData);
    svsList = zeros(nTrain,1); %用来记录支持向量的个数，非0即支持向量
    for ii = 0:nuclass-2
        for jj = (ii+1):nuclass-1
            nn = nn + 1;
            iclass = class(ii+1);
            jclass = class(jj+1);
            iindex = find(trainLabel==iclass);
            %one v one
            jindex = find(trainLabel==jclass);
            itrainLabel = trainLabel(iindex) - trainLabel(iindex) + 1;%one为1
            jtrainLabel = trainLabel(jindex) - trainLabel(jindex) - 1;%others为-1
            itrainData = trainData(:,iindex);
            jtrainData = trainData(:,jindex);
            ijtrainLabel = [itrainLabel,jtrainLabel];
            ijtrainData = [itrainData,jtrainData];
            Fweight = getGraphSVMFuzzyWeight(itrainData,itrainLabel,jtrainData,jtrainLabel,t,k);
            options=optimset;
            options.LargerScale='off';
            options.Display='off';
            
            n=length(ijtrainLabel);
            H=(ijtrainLabel'*ijtrainLabel).*kernel(ijtrainData,ijtrainData,kertype);
            f=-ones(n,1);
            A=[];
            b=[];
            Aeq=ijtrainLabel;
            beq=0;
            lb=zeros(n,1);
            ub=C*Fweight;
            a0=zeros(n,1);
            tic;
            [a,fval,eXitflag,output,lambda]=quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
            %求b
            [svm,sv_label] = calculate_rho(a,ijtrainData',ijtrainLabel',C,kertype);
            svsList(sv_label(:,:))=1;%单次支持向量
            
            svm.a=a(sv_label);
            svm.Xsv=ijtrainData(:,sv_label);
            svm.Ysv=ijtrainLabel(sv_label);
            trainTime = trainTime + toc;
            tic;
            result=svmTest_multiclass(svm,testData,kertype);
            testTime = testTime + toc;
            %testYList(nn,:) = result.score;
            indPreYmin = find(result.score<0);
            testYList((jj+1),indPreYmin) =  testYList((jj+1),indPreYmin) + 1;
            indPreYmax = find(result.score>0);
            testYList((ii+1),indPreYmax) =  testYList((ii+1),indPreYmax) + 1;
        end
    end
    [maxLabel,maxIndex] = max(testYList);
    testY = class(maxIndex);
    %得到精度
    preY = testY;
    %Acc = size(find(testLabel==preY))/size(testLabel);
    if imbanlance ~=2
        Acc = Gmean(preY,testLabel);
    elseif imbanlance == 2
        Acc = Fmean(preY,testLabel);
    end
    svm.svnum = length(find(svsList==1));
    SVs = SVs + svm.svnum;
    proList = softmax(testYList);
end

end

