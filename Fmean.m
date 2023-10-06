function [fmean] = Fmean(Y,testLabel)
%Y n*1 class-imbalance Gmean
%   此处显示详细说明
class = unique(testLabel);
nuclass = length(class);
class1 = class(1);%Postive
if nuclass >1
    class2 = class(2);%Negative
end
ind1 = find(Y == class1);
TPn = 0;
for i = 1:length(ind1)
    if testLabel(ind1(i)) == class1
        TPn = TPn + 1;
    end
end
TP = TPn/length(find(testLabel == class1));

if nuclass ==1
    fmean = TP;
    return;
end

ind2 = find(Y == class2);
FNn = 0;
for i = 1:length(ind2)
    if testLabel(ind2(i)) == class2
        FNn = FNn + 1;
    end
end
FN = FNn/length(find(testLabel == class2));
fmean = (TP+FN)/2;
end

