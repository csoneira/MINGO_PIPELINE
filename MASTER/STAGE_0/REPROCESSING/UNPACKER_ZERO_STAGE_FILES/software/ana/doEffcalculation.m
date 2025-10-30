function [outputVars] = doEffcalculation(inputVars,conf)


numberOfPlanes = conf.ana.param.strips.planes;
activePlane    = conf.ana.param.strips.ActivePlanes;
Z              = conf.ana.param.strips.planesZPos;
XRange         = conf.ana.param.strips.XRange;
YRange         = conf.ana.param.strips.YRange;



eff            = nan(numberOfPlanes,1);
XY_hit         = zeros(length(XRange) - 1,length(YRange) - 1,numberOfPlanes);
XY_det         = zeros(length(XRange) - 1,length(YRange) - 1,numberOfPlanes);


pInd = [2 3 4;1 3 4;1 2 4; 1 2 3];

for P = 1:numberOfPlanes
    if(activePlane(P) == 1)

        X0 = inputVars{P}{8};X1 = inputVars{pInd(P,1)}{8};X2 = inputVars{pInd(P,2)}{8};X3 = inputVars{pInd(P,3)}{8};
        Y0 = inputVars{P}{9};Y1 = inputVars{pInd(P,1)}{9};Y2 = inputVars{pInd(P,2)}{9};Y3 = inputVars{pInd(P,3)}{9};
        Z0 =          Z(P);   Z1 =          Z(pInd(P,1));   Z2 =          Z(pInd(P,2));   Z3 =          Z(pInd(P,3));
        eff0 = zeros(size(X0));
        
        indxHit = find(~isnan(X1) & ~isnan(X2) & ~isnan(X3) & ~isnan(Y1) & ~isnan(Y2) & ~isnan(Y3));

        %%% extrapolated vars from the fit in the plane under study
        X0_ = nan(size(X1));Y0_ = nan(size(X1));Z0_ = nan(size(X1));eff0_ = nan(size(X1));
        for i=1:length(indxHit)
            indx_ = indxHit(i);
            [X0_(indx_), Y0_(indx_), Z0_(indx_)] = extapolate3D_fit( [X1(indx_) Y1(indx_) Z1;X2(indx_) Y2(indx_) Z2;X3(indx_) Y3(indx_) Z3],Z(P));
            if(~isnan(X0(indx_)) & ~isnan(Y0(indx_)))
                    eff0_(indx_) = 1;
            end
        end
         
        indxDet = find(~isnan(eff0_));

        eff(P)   = nansum(eff0_)/length(indxHit);


        [XY_hit(:,:,P),~,~,~] = strips2Dplots(X0_(indxHit), Y0_(indxHit),X0_*0,XRange,YRange,0);XY_hit(:,:,P) = XY_hit(:,:,P)';
        [XY_det(:,:,P),~,~,~] = strips2Dplots(X0_(indxDet), Y0_(indxDet),X0_*0,XRange,YRange,0);XY_det(:,:,P) = XY_det(:,:,P)';

    end
end

indx = find(~isnan(X0) & ~isnan(Y0) & ~isnan(X1) & ~isnan(Y1) & ~isnan(X2) & ~isnan(Y2) & ~isnan(X3) & ~isnan(Y3));
R1 = nan(length(indx),2);R2 = nan(length(indx),2);R3 = nan(length(indx),2);R4 = nan(length(indx),2);

for i=1:length(indx)
    [R1(i,:), R2(i,:), R3(i,:), R4(i,:)] = residuals3D_fit( [X0(indx(i)) Y0(indx(i)) Z(1);X1(indx(i)) Y1(indx(i)) Z(2);X2(indx(i)) Y2(indx(i)) Z(3);X3(indx(i)) Y3(indx(i)) Z(4)]);
end

outputVars = {eff, XY_hit, XY_det, [R1 R2 R3 R4]};

return