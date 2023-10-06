function [Fweight] = getGraphSVMFuzzyWeight(trainData1,trainLabel1,trainData2,trainLabel2,t,k)
%trainData dim*n
if isempty(k)
    k = 1;
end
[~,N1] = size(trainData1);
[~,N2] = size(trainData2);
kdistances1 = zeros(N1, 1);
kdistances2 = zeros(N2, 1);
K1 = round(sqrt(N1)*k);
K2 = round(sqrt(N2)*k);
%%%%%%%%class one%%%%%%%
if(N1 == 1)
    Fweight1 = 1;
else
X2 = sum( trainData1.^2 , 1 );
distance = repmat( X2 , N1 , 1 ) + repmat( X2' , 1 , N1 ) - 2 * trainData1' * trainData1;  % 2�����������
maxDis = max(max(distance));
distance = distance./maxDis;
%d1 = exp( - distance / (2*t*t) );                  % �Ⱥ˺���
d1 = exp( - distance / t );
[ sorted , index ] = sort(distance);         % sort �Ƕ��н������򣬷���sortedÿ����������index����
neighborhood = index(2:(1+K1),:);             % K ���ڣ�neiborhood(K,N)
% STEP2: Construct similarity matrix W
W = zeros(N1, N1);
% way2 : K �ڽ���ֵ��˹����
for ii = 1 : N1
    W( ii , neighborhood(:, ii) ) = d1( ii , neighborhood(:, ii) );
    W( neighborhood(:, ii) , ii ) = d1( ii , neighborhood(:, ii) )';
end
% STEP 3: COMPUTE EMBEDDING FROM EIGENVECTS OF L
D = sum( W , 2 );
Fweight1 = D./sum(D);  
end
%%%%%%%class two%%%%%%%
X2 = sum( trainData2.^2 , 1 );
distance = repmat( X2 , N2 , 1 ) + repmat( X2' , 1 , N2 ) - 2 * trainData2' * trainData2;  % 2�����������
maxDis = max(max(distance));
distance = distance./maxDis;
d1 = exp( - distance / t );                  % �Ⱥ˺���
[ sorted , index ] = sort(distance);         % sort �Ƕ��н������򣬷���sortedÿ����������index����
neighborhood = index(2:(1+K2),:);             % K ���ڣ�neiborhood(K,N)
% STEP2: Construct similarity matrix W
W = zeros(N2, N2);
% way2 : K �ڽ���ֵ��˹����
for ii = 1 : N2
    W( ii , neighborhood(:, ii) ) = d1( ii , neighborhood(:, ii) );
    W( neighborhood(:, ii) , ii ) = d1( ii , neighborhood(:, ii) )';
end
% STEP 3: COMPUTE EMBEDDING FROM EIGENVECTS OF L
D = sum( W , 2 );
Fweight2 = D./sum(D);

Fweight = [Fweight1;Fweight2];
end

