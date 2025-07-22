function [XY,XY_Qmean,XY_Qmedian,XY_ST,XY_Q] = strips2DplotsAdvance(X,Y,Q,binsX,binsY,STLevel)

XY         = zeros(length(binsX)-1,length(binsY)-1);
XY_Qmean   = zeros(length(binsX)-1,length(binsY)-1);
XY_Qmedian = zeros(length(binsX)-1,length(binsY)-1);
XY_ST      = zeros(length(binsX)-1,length(binsY)-1);
XY_Q       = cell(length(binsX)-1,length(binsY)-1);

for i=1:length(binsX)-1
    for j= 1:length(binsY)-1
        I = find(X > binsX(i) & X <= binsX(i+1) & Y > binsY(j) & Y <= binsY(j+1));
        XY(i,j)        = length(I);
        XY_Qmean(i,j)  = nanmean(Q(I));
        %XY_Qmedian(i,j)= nanmedian(Q(I));
        XY_Qmedian(i,j) = 0;
        XY_ST(i,j)     = length(find(Q(I) > STLevel));
        XY_Q{i,j}      = Q(I);
    end
end

I = isnan(XY_Qmean);XY_Qmean(I) = 0;
%I = isnan(XY_Qmedian);XY_Qmedian(I) = 0;