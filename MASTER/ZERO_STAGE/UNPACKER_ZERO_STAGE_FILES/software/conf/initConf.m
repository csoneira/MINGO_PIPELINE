%% Check system
[status, result] = system('hostname');
result = strtrim(result);  % Remove trailing newline or whitespace

% Retrieve SYSTEMNAME from environment variable
env_system = getenv('RPCSYSTEM');
if isempty(env_system)
    env_system = 'mingo01'; % Fallback if not defined
end

if strcmp(result, 'manta')
    OS = 'linux';
    HOSTNAME    = 2;
    SYSTEMNAME  = env_system;
    % Software location
    HOME        = ['/home/alberto/gate/localDocs/lip/daqSystems/' SYSTEMNAME '/'];
    % System data structure
    SYS         = [HOME 'system/'];
    INTERPRETER = 'matlab';
else
    more off
    warning('off');
    OS = 'linux';
    HOSTNAME    = 1;
    SYSTEMNAME  = env_system;
    % Software location
    HOME        = '/home/mingo/DATAFLOW_v3/MASTER/ZERO_STAGE/UNPACKER_ZERO_STAGE_FILES/';
    % System data structure
    SYS         = [HOME 'system/'];
    INTERPRETER = 'octave';
end

