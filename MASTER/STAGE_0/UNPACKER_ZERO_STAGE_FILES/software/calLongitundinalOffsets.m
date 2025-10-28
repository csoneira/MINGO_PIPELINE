%% Configuration
clear all;close all;

run('./conf/initConf.m');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
run([HOME 'software/conf/loadGeneralConf.m']);


%% Load configuration
conf = initSystem();
conf = loadConfiguration({conf,HOSTNAME,SYSTEMNAME,HOME,SYS,INTERPRETER,OS});    
b = conf.bar;
% 
%
write2log(conf.logs,'','   ','syslog',OS);write2log(conf.logs,'','   ','syslog',OS);
message2log = ['***************************************************************'];
disp(message2log);
write2log(conf.logs,message2log,'   ','syslog',OS);
message2log = ['*** Starting pedestalCalculation.'];
disp(message2log);
write2log(conf.logs,message2log,'   ','syslog',OS);

%oldFile4Offsets = '/home/alberto/gate/localDocs/lip/Projects-RD/STRATOS/offLinesystem/stratos1/system/lookUpTables/YOffsets_21078000211.mat';
file4Offsets    = [conf.ana.calibration.longitudinalY.path 'outMerge4LongitudinalOffset.mat'];



numberOfPlanes = conf.ana.param.strips.planes;
activePlanes   = conf.ana.param.strips.ActivePlanes;
strips         = conf.ana.param.strips.strips;
YCenters       = nan(strips,numberOfPlanes);
binSize        = 0.01;
th             = 100;

load(file4Offsets);





for plane = 1:numberOfPlanes
    if(activePlanes(plane))
         X = XY(:,1 + 2*((plane)-1));
         Y = XY(:,2 + 2*((plane)-1));
        for i=1:strips
            disp(['Strip ' num2str(strips) ' in plane ' num2str(plane)]);
            try
                I = find(X == i);
                [x,y] = histf(Y(I),-3:binSize:3);
                stairs(y,x);
                bins = find( x > th);
                %n(find(x > median(x(find(x)))*0.4))
                C(i) = y(bins(end)) - ((y(bins(end)) -  y(bins(1)))/2);
                stripLength(i) = y(bins(end)) - y(bins(1))
                hold on
                [x,y] = histf(Y(I)-C(i),-3:binSize:3);
                stairs(y,x)
                %keyboard
                hold off
            catch
            end
        end
        YCenters(:,plane) =  C';
    end
end




%save([conf.ana.calibration.longitudinalY.path 'YOffSet_2023_06_05.mat'],'YCenters');




