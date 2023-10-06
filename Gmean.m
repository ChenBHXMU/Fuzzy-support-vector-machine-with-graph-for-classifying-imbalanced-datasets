function [gmean] = Gmean(Y,testLabel)
%Y n*1 class-imbalance Gmean
%   此处显示详细说明
%Y n*1 class-imbalance Gmean  多分类
%   此处显示详细说明
class = unique(testLabel);
nuclass = length(class);

acc = 1;
for i = 1:nuclass
    ind1 = find(testLabel == class(i));
    ind2 = find(Y == class(i));
    if ~isempty(ind2)
        C = intersect(ind1,ind2);
        acc1 = length(C)./length(ind1);
        acc = acc.*acc1;
    else
        gmean = 0;
        return;
    end
end
gmean = acc^(1/nuclass);
end

