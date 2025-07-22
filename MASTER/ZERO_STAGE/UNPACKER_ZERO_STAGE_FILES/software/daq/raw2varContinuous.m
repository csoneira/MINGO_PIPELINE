function raw2varContinuous(inputPath,outputPath,TRBs,lookUpTables,keepRawFiles,interpreter,logs,OS)

try
    
    %%    GO
    numberOfTRBs = size(TRBs,2);
    
    for TRBn=1:numberOfTRBs
        %%   Check the configuration on the conf file and take the FPGAs active, and the list of files
        %%%%  This is the possition on the list of FPGAs
        TDCIndex = find((strcmpi('TDC',TRBs(TRBn).FPGAs(:,2)) | strcmpi('TDC-CTS',TRBs(TRBn).FPGAs(:,2))) & strcmpi('TRUE',TRBs(TRBn).FPGAs(:,3)) & strcmpi('TRUE',TRBs(TRBn).FPGAs(:,4)));
        ADCIndex = find((strcmpi('ADC',TRBs(TRBn).FPGAs(:,2)) | strcmpi('ADC-GBE',TRBs(TRBn).FPGAs(:,2))) & strcmpi('TRUE',TRBs(TRBn).FPGAs(:,3)) & strcmpi('TRUE',TRBs(TRBn).FPGAs(:,4)));
        CTSIndex = find( strcmpi('CTS',TRBs(TRBn).FPGAs(:,2))                                             & strcmpi('TRUE',TRBs(TRBn).FPGAs(:,3)) & strcmpi('TRUE',TRBs(TRBn).FPGAs(:,4)));
        
       
        
        %%%% This is the corresponding Id C000, A001, ...
        TDCIDs  = TRBs(TRBn).FPGAs(TDCIndex,1);
        ADCIDs  = TRBs(TRBn).FPGAs(ADCIndex,1);
        CTSIDs  = TRBs(TRBn).FPGAs(CTSIndex,1);
        HEADIDs = {'HEAD'};

        allFPGAsIDs = [TDCIDs; ADCIDs; CTSIDs; HEADIDs];
        
        %%%   Check if the number of files are consistent. All FPGAS active must have the same number of files.
        numberOfFiles = [];
        for j=1:size(allFPGAsIDs,1)
            d = dir([inputPath '*' TRBs(TRBn).centralFPGA '_' allFPGAsIDs{j} '*']);
            numberOfFiles = [numberOfFiles; size(d,1)];
        end
        
        if(size(numberOfFiles,2) ~= 1)
            disp('Number of files is not equal => somethng is wrong');
            keyboard
        end
        
        if(sum(numberOfFiles) == 0)
            message2log = ['No files to read for TRB ' TRBs(TRBn).centralFPGA ' . Skipping'];
            disp(message2log);
            write2log(logs,message2log,'   ','syslog',OS);
            continue
        end
        clear numberOfFiles d
        
        %%%% Check the files existing in the folder.
        allMatFiles    = dir([inputPath '*.mat']);allMatFiles    = strvcat(allMatFiles.name);
        
        for i=1:size(allMatFiles,1)
            allMatFilesCell{i} = allMatFiles(i,1:(min(find(allMatFiles(i,:) == '_'))-1));
        end
        
        
        uniqueMatFiles = unique(allMatFilesCell,'rows');
        %uniqueMatFiles = unique(allMatFiles(:,1:end-23),'rows');
        
        
              
        %% Process files in three loops, files, FPGAs, parts
        for files = 1:size(uniqueMatFiles,2)
            fileName = uniqueMatFiles{files};
            message2log = ['Processing file ' fileName];disp(message2log);write2log(logs,message2log,'   ','syslog',OS);
            if strcmp('matlab',interpreter)
                if(~exist([outputPath fileName '.mat'],'file'))
                    save([outputPath fileName '.mat'],'TRBs');
                else
                    save([outputPath fileName '.mat'],'TRBs','-append');
                end
            elseif strcmp('octave',interpreter)
                if(~exist([outputPath fileName '.mat'],'file'))
                    save([outputPath fileName '.mat'],'TRBs','-mat7-binary');
                else
                    save([outputPath fileName '.mat'],'TRBs','-append','-mat7-binary');
                end
            else
            end
            
            %% Process TDC files
            for Ids = 1:size(TDCIDs,1)
                %%% Create the temp vars
                eval(['F_' TDCIDs{Ids,:} ' = [];']);
                
                %%% Process the part files and move to destination
                filesTDC = dir([inputPath fileName '_' TRBs(TRBn).centralFPGA '_' TDCIDs{Ids,:} '*']);
                for filesPart = 1:size(filesTDC,1)
                    %processTRBTime(filesTDC(filesPart).name, conf.TRBs(TRBn).FPGAs{TDCIndex(Ids),5},conf)
                    %               fileName                , timeCalibrationParameters             ,conf
                    timeCalPar = TRBs(TRBn).FPGAs{TDCIndex(Ids),5};
                    time = processTRBTDC(filesTDC(filesPart).name,timeCalPar,inputPath);
                    
                    eval(['F_' TDCIDs{Ids,:} '    = [F_' TDCIDs{Ids,:} '; time];']);
                    
                    %%% Move files
                    if(keepRawFiles == 1)
                        system(['mv ' inputPath filesTDC(filesPart).name ' ' inputPath '../done/']);
                    else
                        system(['rm ' inputPath filesTDC(filesPart).name]);
                    end
                    clear timeCalPar time
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                end
            end
            
            %% Process ADC files
            for Ids = 1:size(ADCIDs,1)
                %%% Create the temp vars
                eval(['F_' ADCIDs{Ids,:} ' = [];']);
                
                %%% Process the part files and move to destination
                filesADC = dir([inputPath fileName '_' TRBs(TRBn).centralFPGA '_' ADCIDs{Ids,:} '*']);
                for filesPart = 1:size(filesADC,1)
                    
                    Q = processTRBADC_2(filesADC(filesPart).name,inputPath);
                                                          
                    eval(['F_' ADCIDs{Ids,:} ' = cat(3,F_' ADCIDs{Ids,:} ',Q);']);
                                                            
                    %%% Move files
                    if(keepRawFiles == 1)
                        system(['mv ' inputPath filesADC(filesPart).name ' ' inputPath '../done/']);
                    else
                        system(['rm ' inputPath filesADC(filesPart).name]);
                    end
                    clear Q
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                end
            end
            
            %% Process HEAD files
            for Ids = 1:size(HEADIDs,1)
                %%% Create the temp vars
                eval(['a_' HEADIDs{Ids,:} ' = [];']);eval(['b_' HEADIDs{Ids,:} ' = [];']);eval(['c_' HEADIDs{Ids,:} ' = [];']);eval(['d_' HEADIDs{Ids,:} ' = [];']);
                
                %%% Process the part files and move to destination
                filesHEAD = dir([inputPath fileName '_' TRBs(TRBn).centralFPGA '_' HEADIDs{Ids,:} '*']);
                for filesPart = 1:size(filesHEAD,1)
                    
                    
                    [triggerType, filePosition,fileNames,EBtime] = processTRBHEAD(filesHEAD(filesPart).name,inputPath);
                    
                    eval(['a_' HEADIDs{Ids,:} '    = [a_' HEADIDs{Ids,:} '; filePosition];']);
                    eval(['b_' HEADIDs{Ids,:} '    = [b_' HEADIDs{Ids,:} '; fileNames];']);
                    eval(['c_' HEADIDs{Ids,:} '    = [c_' HEADIDs{Ids,:} '; EBtime];']);
                    eval(['d_' HEADIDs{Ids,:} '    = [d_' HEADIDs{Ids,:} '; triggerType];']);

                    %%% Move files
                    if(keepRawFiles == 1)
                        system(['mv ' inputPath filesHEAD(filesPart).name ' ' inputPath '../done/']);
                    else
                        system(['rm ' inputPath filesHEAD(filesPart).name]);
                    end
                    clear filePosition fileNames EBtime triggerType
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                end
            end
            
                           
            %% Apply lookUpTables and Save
            %%%
            run(lookUpTables{TRBn});
            
            for varIDs = 1:size(lookUpInstructions,1)
                eval([lookUpInstructions{varIDs,1} ' = ' lookUpInstructions{varIDs,2} ';']);
                if strcmp('matlab',interpreter)
                    save([outputPath fileName '.mat'],lookUpInstructions{varIDs,1},'-append');
                elseif strcmp('octave',interpreter)
                    save([outputPath fileName '.mat'],lookUpInstructions{varIDs,1},'-append','-mat7-binary');
                else
                end
            end
            message2log = ['Processing file done'];disp(message2log);write2log(logs,message2log,'   ','syslog',OS);
        end
    end
catch exception
       
        for i = 1:length(exception.stack)
            message2log = ['Error in: ' exception.stack(i).file ' line ' num2str(exception.stack(i).line)];
            disp(message2log);
            write2log(logs,message2log,'   ','syslog',OS);
            write2log(logs,message2log,'   ','criticallog',OS);
        end
    
end
return