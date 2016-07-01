% Load files from NEST to plot etc
clear all; clc; close all hidden

for o1=1:1 % Generate some fake volts/spikes in NEST
    
% % Copy into Nest to generate the inputs there

% # Fake readouts for importing into matlab
% f_input=nest.Create("poisson_generator", 1,  {'rate': 10000., 'start': 0., })
% f_neuron=Create('iaf_neuron', 1)
% f_volt=nest.Create("voltmeter", params={"to_file": True, "withtime": True})
% f_sd=nest.Create("spike_detector", params={"to_file": True, "withtime": True})
% nest.CopyModel("static_synapse","excitatory",{"weight":100., "delay":1.})
% nest.Connect(f_input, f_neuron, model='excitatory')
% nest.Connect(f_volt, f_neuron, model='excitatory')
% nest.Connect(f_neuron, f_sd, model='excitatory')
% nest.Simulate(500.0)
% events=nest.GetStatus(f_volt,'events')[0]
% print "The last ten timesteps for the Vm are",
% print events['V_m'][-10:]
% 
% print " \n"*5
% print "Did we see spikes? If so, they happened at : ",
% spikes = nest.GetStatus(f_sd,'events')[0]
% print spikes['times']


    
end

request.voltmeters={
    'CA1'               'vlt_ca1'; 
    'CA3'               'vlt_ca3';
    'Amy'              'vlt_amy';
    };

request.spikedetectors={
    'CA1'               'sd_ca1'; 
    'CA3'               'sd_ca3';
    'Amy'              'sd_amy';
    'PoissonGen'    'sd_input';
    };
                     
%% Preprocessing

request.nodes=unique([request.spikedetectors(:,1); request.voltmeters(:,1)]);

for o1=1:1 % Readout column specifications
    vc.time=2;
    vc.v=3;
    
    
end
for o1=1:1  % Load readout data (d_read)
    d_read=[request.spikedetectors(:,1) cell(length(request.spikedetectors(:,1)), 2)];
    for v=1:size(request.voltmeters,1) % Voltmeters
        d_read{find(strcmp(d_read(:,1),request.voltmeters{v} )), 2}=eval(['load('''  spm_select('List', pwd,  request.voltmeters{v,2} ) ''');']);
    end
    for s=1:size(request.spikedetectors,1) % Spike detectors
        d_read{find(strcmp(d_read(:,1),request.spikedetectors{s} )), 3}  =eval( ['load('''  spm_select('List', pwd, request.spikedetectors{s,2} ) ''');']);
    end
    cd ..
end
request.nodes={'PoissonGen';'CA3';'Amy';'CA1';};
% request.nodes={'CA3';'Amy';'CA1';};
% request.nodes=request.spikedetectors(:,1);

%% Plots

f.figwidth=600;f.figheight=800; f.subplotcols=1; f.subplot_VerHorz=[0.05 0.07]; f.fig_BotTop=[0.1 0.1]; f.fig_LeftRight=[0.15 0.1];
figure('NumberTitle', 'off', 'Position',[200,00,f.figwidth,f.figheight]); set(gcf,'Color',[1 1 1]);
for nn=1:length(request.nodes)
    n=find(strcmp(d_read(:,1), request.nodes{nn}));
%     subtightplot(length(request.nodes),f.subplotcols, (n-1)*f.subplotcols+1,f.subplot_VerHorz,f.fig_BotTop, f.fig_LeftRight);
%     title(d_read{n,1}, 'FontSize', 18); axis off



    % Voltage + Spikes
    subtightplot(length(request.nodes),f.subplotcols, (nn-1)*f.subplotcols+1,f.subplot_VerHorz,f.fig_BotTop, f.fig_LeftRight);
    switch d_read{n}
        case 'PoissonGen'
            line([d_read{n,3}(:,2)   d_read{n,3}(:,2)],  [0 10], 'color', 'r')
        otherwise
            plot(d_read{n,2}(:,2),  d_read{n,2}(:,3));
            if isempty(d_read{n,3})==0 % Add spikes if any
                hold on; line([d_read{n,3}(:,2)   d_read{n,3}(:,2)],  [min(d_read{n,2}(:,3)) max(d_read{n,2}(:,3))], 'color', 'r')
            end
    end
    title(d_read{n}, 'FontSize', 18)
    ylim('auto')

end

