% s_IAFnetwork
%   McCullock Pitts - integrate multiple inputs, output neuron spikes/doesns (Boolean)
%   Need to define weights (W) and threshold T of output neuron, compare
%   weighted sum to threshold
clear all; close all hidden; clc

% This script is just a sketch tutorial laying out how a network might be
% built in MATLAB. Because Python/NEST is (a) WAY faster and (b) packed
% with functionality, I've built my model in Py instead. 


%% 5 neurons 

dothis=0;
if dothis
    
    for o1=1:1
        % Setup
        n_neur=5;
        th=0.9;
        t_stop=1; % in seconds
        dt=0.01; % t_stop steps
        
        % Initialize
        u=rand(5,t_stop./dt)>0.5;
        w=rand(5,1);
        
        counter=0;
        for i=0:dt: (t_stop-dt)
            counter=counter+1;
            state(counter)=sum( u(:, counter).*w) > th;
        end
        
    end

end

%% Population - balanced excit-einhibit (Carl's balanced EI network)

dothis=0;
if dothis
    for o1=1:1
        N=100;
        NE=80/100 * N;  % Inherent connectivity
        NI=20/100 * N;
        dt=0.01; % t_stop steps
        t_stop=1; % in seconds
        
        
        
        % Wee=rand(NE,NE) * 0.2;  % Exc-Exc connectivity matrix. Multiply by connection strengths HERE
        % Wie=rand(NI,NE) * 0.2;    %  Columns=input, Row=output
        % Wei=rand(NE,NI) * 0.2;
        % Wii=rand(NI,NI) * 0.2;
        %
        k=5; % Make network sparse
        Wee=rand(NE,NE) .* (rand(NE,NE) < k/NE);
        Wie=rand(NI,NE) .* (rand(NI,NE) <k/NE);
        Wei=rand(NE,NI) .* (rand(NE,NI) <k/NI);
        Wii=rand(NI,NI) .* (rand(NI,NI) <k/NI);
        
        X=round(rand(NE,1));   % initially, boolean spiking/not
        Y=round(rand(NI,1));
        
        counter=0;
        U=0;  % Input
        TE=0.5;  % threshold
        TI= 0.8;
        for i=0:dt:(t_stop-dt)
            counter=counter+1;
            
            X= Wee*X - Wei * Y + U;  % Excitatory units receive excit from other excits, inhib from inhib units, and the overall input (u)
            X=X>TE; % Apply threshold
            Y=Wie* X - Wii* Y; % Inhibitory inputs are not directly driven (no u)
            Y=Y>TI;
            H(counter, :)=[X;Y];
        end
        
        figure, imagesc(H'), colorbar, title('Rasta')
        req={'Wee';'Wie';'Wei';'Wii'}; figure
        for r=1:length(req)
            subplot(2,2,r);
            eval(['imagesc(' req{r} ');'])
            eval(['title(''' req{r} ''');'])
            colorbar
        end
        
    end
    
%  Further steps    
%     % Steps from previous: 
%     a) input connections (exactly k per unit)
%     b) input spike trains
%     c) normalize excit and inhib per neuron

end

%% IAF neuron
% voltage accmulates (driven by inputs and recurrents), spikes when it hits
% the threshold, then resets after

% % Constants
%     T % threshold
%     L % Leak 
%     R % Reset
% % Dynamics
%     V : membrane voltage
%     O: spike status

% Setup
% (1) Vi = -L.*Vi(t) + Wi.*X(t)  + sum( Wij.* Oj);    % Voltage == 
%     o=spiking status of neuron j (connected to neuron i)
%     X= driving input, Wi=weights from driving input to neuron i
% (2) if Vi(t) > Ti , Oi=1;   else Oi=0; end % Apply threshold to current voltage
%
% Get derivative of equation Vi



for t=1:timesteps
    
    % Update voltage
    
    % Check threshold (V>T), otherwise set to 0
    
    % 
    
end




