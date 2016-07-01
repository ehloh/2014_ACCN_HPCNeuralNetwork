import sys
sys.path.append("/Users/EleanorL/NEST/nest24/ins/lib/python2.7/site-packages")
import nest
import nest.raster_plot
import nest.voltage_trace
import numpy as np
import matplotlib.pyplot as plt

dt=0.1
nest.ResetKernel()
nest.SetKernelStatus({"overwrite_files": True, 'data_path': '/Users/EleanorL/Dropbox/SANDISK/12_ACCN/1_Project/2_Simulated_data' , "resolution":dt})

def build_network(modinput): # (input_noiserate, weight_noiseinput):

    # # Free parameters ######  
    # input_noiserate=modinput['input_noiserate']
    # weight_noiseinput=modinput['weight_noiseinput']
    # c_inhib_weightmultiplier=modinput['c_inhib_weightmultiplier']
    # c_fudgeweights=modinput['c_fudgeweights']
    # c_ca3_2inhib_bonus=modinput['c_ca3_2inhib_bonus']
    """ Specifid parameter values (from scan) """
    # input_noiserate=19800.    # input rate & weight = 19800 & 9 balances CA3 at 22Hz (excit and inhib oscillate)
    # weight_noiseinput=9.
    input_noiserate=19000.      # [Background] input rate & weight= 19000 & 9 will get CA3 down to <8 Hz (excit and inhib oscillate)
    weight_noiseinput=9.
    # input_noiserate=20400.      # input rate & weight= 20500 & 8.25 will get CA3 down to <10 Hz (excit and inhib oscillate)
    # weight_noiseinput=8.25
    # input_noiserate=20550.      # input rate & weights arbitrary
    # weight_noiseinput=8.25
    c_inhib_weightmultiplier= 90./71.
    c_fudgeweights=10.5
    c_ca3_2inhib_bonus= 3.
    c_ca3_recur_bonus= -0.2
    c_ca3_2ca1e_bonus=0.5
    c_ca1_e2i_bonus=-0.3
    c_ca1_i2e_pbonus= -0.3
    c_ca1_i2i_pbonus= -0.3

    # # # Artificially jacked up for the taster plots!!
    # input_noiserate=19800.
    # weight_noiseinput=9.
    
    # input_noiserate=18900.    # FAKE TEST IF GRID IS OK
    # weight_noiseinput=8.75
    # c_fudgeweights=7.5

    """ Builds the network """
    nest.ResetKernel()
    nest.SetKernelStatus({"overwrite_files": True, 'data_path': '/Users/EleanorL/Dropbox/SANDISK/12_ACCN/1_Project/2_Simulated_data' , "resolution":dt})

    # Node setup
    prop_pyramidpop_e=0.9
    type_ca3='iaf_psc_exp'  # CA3
    n_ca3=1000
    n_ca3e=int(round(n_ca3*prop_pyramidpop_e))
    n_ca3i=int(round(n_ca3*(1-prop_pyramidpop_e)))
    ns_ca3e_mempot= - 65.3
    ns_ca_inhib_mempot= - 62.
    type_ca1=type_ca3       # CA1 
    n_ca1=n_ca3
    n_ca1e=int(round(n_ca1*prop_pyramidpop_e))
    n_ca1i=int(round(n_ca1*(1-prop_pyramidpop_e)))
    ns_ca1e_mempot= - 62.6
    n_amye=n_ca3e           # AMY   
    type_amy=type_ca3
    cs_timecon_ex2=2.   # Synapse time constants (CA3/CA1 only)
    cs_timecon_inhib2inhib=2.
    cs_timecon_inhib2excit=7.
    c_ca3_2ca1i_bonus = -0.3


    # Connectivity = [ConnectivityProbability  Weights]  Intrinsic connex from Taxidis 2011 HPC + distortions
    c_ca3_recur= [0.1062,  (1.+c_ca3_recur_bonus)*c_fudgeweights]    # Intra-CA3
    c_ca3_2inhib= [0.0988,  (1.6+c_ca3_2inhib_bonus)*c_fudgeweights]
    c_ca3_inhib2excit= [0.713, 1.2*c_fudgeweights* -1 * c_inhib_weightmultiplier]
    c_ca1_2inhib= [0.3111,  (1.3+c_ca1_e2i_bonus)*c_fudgeweights]    # Intra-CA1
    c_ca1_inhib2excit=[(0.8397+c_ca1_i2e_pbonus),  0.5*c_fudgeweights*-1 * c_inhib_weightmultiplier]
    c_ca1_inhib2inhib=[(0.9089+c_ca1_i2i_pbonus),  0.5*c_fudgeweights*-1 * c_inhib_weightmultiplier]
    c_ca3_2ca1= [0.1821,  (0.1+ c_ca3_2ca1e_bonus)*c_fudgeweights]  # CA3 --> CA1 (Schaffer)
    c_ca3_2ca1i= [0.1825,  (0.4 + c_ca3_2ca1i_bonus) *c_fudgeweights]  # CA3 --> CA1 interneurons
    # Weights from here are arbitrary !!! 
    c_ca3_2amye=[0.95,  0.6*c_fudgeweights]   # This, with poisson weights to amygdaala of 0.8 work FINE.
    c_amye_2ca1e=[0.55,  1.5*c_fudgeweights]





    # TRY THIS 
    c_ca3_2amye=[0.8,  0.2*c_fudgeweights] 


    # # Dynamics
    theta_amp=7.
    theta_freq= 7.5  # theta freq: 6-10 Hz
    theta_offset=0.
    theta_phase=0.
    weight_thetainput=0.8
    theta_dc=0.
    
    # ################################################################################
    # Create nodes + Readouts
    nest.CopyModel("static_synapse","readout", {"weight":1., "delay": dt})  
    input_noise=nest.Create("poisson_generator", 1, {'rate': input_noiserate, 'start': 0., } )
    sd_noiseinput=nest.Create("spike_detector") # , params={"to_file": True, "withtime": True, "label":'sd_ca3'})  
    nest.Connect(input_noise, sd_noiseinput, model='readout')
    input_theta=nest.Create("sinusoidal_poisson_generator", 1, {'dc': theta_dc, 'ac': theta_amp,  'freq': theta_freq} )
    sd_theta=nest.Create("spike_detector") # , params={"to_file": True, "withtime": True, "label":'sd_ca3'})  
    nest.Connect(input_theta, sd_theta, model='readout')
    input_active=nest.Create("sinusoidal_poisson_generator", 1, {'dc': theta_dc, 'ac': theta_amp,  'freq': theta_freq} )
    sd_theta=nest.Create("spike_detector") # , params={"to_file": True, "withtime": True, "label":'sd_ca3'})  
    nest.Connect(input_theta, sd_theta, model='readout')
    # nest.Create('spike_generator',1, {'spike_times': float(range(1,100,1)})
    # sd_active=nest.Create("spike_detector") # , params={"to_file": True, "withtime": True, "label":'sd_ca3'})  
    # nest.Connect(input_active, sd_active, model='readout')
    pope_ca3=nest.Create(type_ca3, n_ca3e, {'tau_syn_ex': cs_timecon_ex2})    # CA3 ------------------------------
    popi_ca3=nest.Create(type_ca3, n_ca3i, {'tau_syn_ex': cs_timecon_inhib2excit, 'tau_syn_in': cs_timecon_inhib2inhib})    
    noise = np.random.normal(nest.GetStatus(pope_ca3)[2]['V_m'], 2, len(pope_ca3))
    for n in range(len(pope_ca3)):
        nest.SetStatus([pope_ca3[n]], {'V_m':noise[n]})
    sd_ca3=nest.Create("spike_detector") # , params={"to_file": True, "withtime": True, "label":'sd_ca3'})  
    vlt_ca3= nest.Create("voltmeter") # , params={"to_file": True, "withtime": True,"label":'vlt_ca3'})  
    sd_ca3i=nest.Create("spike_detector") # , params={"to_file": True, "withtime": True, "label":'sd_ca3i'})  
    vlt_ca3i= nest.Create("voltmeter") # , params={"to_file": True, "withtime": True,"label":'vlt_ca3i'})  
    nest.Connect(vlt_ca3, [pope_ca3[1]], model='readout') 
    nest.ConvergentConnect(pope_ca3, sd_ca3, model='readout')
    nest.Connect(vlt_ca3i, [popi_ca3[1]], model='readout')
    nest.ConvergentConnect(popi_ca3, sd_ca3i, model='readout')
    pope_ca1=nest.Create(type_ca1, n_ca1e, {'tau_syn_ex': cs_timecon_ex2})    # CA1 ------------------------------
    popi_ca1=nest.Create(type_ca1, n_ca1i, {'tau_syn_ex': cs_timecon_inhib2excit, 'tau_syn_in': cs_timecon_inhib2inhib})    
    sd_ca1=nest.Create("spike_detector") # , params={"to_file": True, "withtime": True, "label":'sd_ca1'})  
    vlt_ca1= nest.Create("voltmeter") # , params={"to_file": True, "withtime": True,"label":'vlt_ca1'})  
    sd_ca1i=nest.Create("spike_detector") # , params={"to_file": True, "withtime": True, "label":'sd_ca1i'})  
    vlt_ca1i= nest.Create("voltmeter") # , params={"to_file": True, "withtime": True,"label":'vlt_ca1i'})  
    nest.Connect(vlt_ca1, [pope_ca1[1]], model='readout') 
    nest.ConvergentConnect(pope_ca1, sd_ca1, model='readout')
    nest.Connect(vlt_ca1i, [popi_ca1[1]], model='readout')
    nest.ConvergentConnect(popi_ca1, sd_ca1i, model='readout')
    pope_amy=nest.Create(type_amy, n_amye)   # amy ------------------------------
    sd_amy=nest.Create("spike_detector") # , params={"to_file": True, "withtime": True, "label":'sd_amy'})  
    vlt_amy= nest.Create("voltmeter") # , params={"to_file": True, "withtime": True,"label":'vlt_amy'})  
    nest.Connect(vlt_amy, [pope_amy[1]], model='readout') 
    nest.ConvergentConnect(pope_amy, sd_amy, model='readout')

    """ These weights work (log) """
    # c_ca3_2ca1= [0.1821,  (0.6)*c_fudgeweights]  # CA3 --> CA1 (Schaffer)
    # c_amye_2ca1e=[0.4,  1.5*c_fudgeweights]
    # c_ca1_2inhib= [0.3111,  (1.3-0.3)*c_fudgeweights]    # Intra-CA1
    # c_ca1_inhib2excit=[0.8397-0.3,  0.5*c_fudgeweights*-1 * c_inhib_weightmultiplier]
    # c_ca1_inhib2inhib=[0.9089-0.3,  0.5*c_fudgeweights*-1 * c_inhib_weightmultiplier]

    """ Connectivity """
    nest.DivergentConnect(input_noise, pope_ca3, weight_noiseinput, dt)  
    # nest.DivergentConnect(input_theta, pope_ca3, weight_thetainput, dt)  

    # Within CA3
    nest.RandomDivergentConnect(pope_ca3, pope_ca3, int(round(c_ca3_recur[0]*n_ca3e)), weight=c_ca3_recur[1], delay=1.)   
    nest.RandomDivergentConnect(pope_ca3, popi_ca3, int(round(c_ca3_2inhib[0]*n_ca3i)), weight=c_ca3_2inhib[1], delay=1.)
    nest.RandomDivergentConnect(popi_ca3, pope_ca3, int(round(c_ca3_inhib2excit[0]*n_ca3e)), weight=c_ca3_inhib2excit[1], delay=1.)    
    
    # CA3 --> CA1
    nest.RandomDivergentConnect(pope_ca3, pope_ca1, int(round(c_ca3_2ca1[0]*n_ca1e)), weight=c_ca3_2ca1[1], delay=1.) 
    nest.RandomDivergentConnect(pope_ca3, popi_ca1, int(round(c_ca3_2ca1i[0]*n_ca1i)), weight=c_ca3_2ca1i[1], delay=1.)
    
    # Within CA1
    nest.RandomDivergentConnect(pope_ca1, popi_ca1, int(round(c_ca1_2inhib[0]*n_ca1i)), weight=c_ca1_2inhib[1], delay=1.)   
    nest.RandomDivergentConnect(popi_ca1, pope_ca1, int(round(c_ca1_inhib2excit[0]*n_ca1e)), weight=c_ca1_inhib2excit[1], delay=1.) 
    nest.RandomDivergentConnect(popi_ca1, popi_ca1, int(round(c_ca1_inhib2inhib[0]*n_ca1i)), weight=c_ca1_inhib2inhib[1], delay=1.) 
    
    # To/From Amygdala
    nest.RandomDivergentConnect(pope_ca3, pope_amy, int(round(c_ca3_2amye[0]*n_amye)), weight=c_ca3_2amye[1], delay=1.)   
    nest.RandomDivergentConnect(pope_amy, pope_ca1, int(round(c_amye_2ca1e[0]*n_ca1e)), weight=c_amye_2ca1e[1], delay=1.)   

    # Noise to nodes
    nest.DivergentConnect(input_noise, popi_ca3, weight_noiseinput*0.1, dt)  
    nest.DivergentConnect(input_noise, pope_amy, weight_noiseinput*0.95, dt)  
    nest.DivergentConnect(input_noise, pope_ca1, weight_noiseinput*0.8, dt)  

    # ################################################################################
    modoutput=locals()
    return modoutput

def scan_weights(time_ms):
    """ Scan across params """
    # time_ms=1000.
    k=1

    # Settings 
    input_noiserates =np.arange(19000.,22000., 500.)
    input_weights =  np.arange(8., 10., 0.5)   
    # fudge_weights=np.arange(0.5, 30., 2.)   
    # c_ca3_2inhib_bonus=np.arange(-5., 5., 0.5)   
    scanpar=[input_noiserates, input_weights]
    # print 'No. of simulations: ', len(scanpar[0])*len(scanpar[1])*len(scanpar[2])*len(scanpar[3])
    # d_spikerate=np.zeros([len(scanpar[0]), len(scanpar[1]), len(scanpar[2]), len(scanpar[3])])    
    # d_spikerate2=np.zeros([len(scanpar[0]), len(scanpar[1]), len(scanpar[2]), len(scanpar[3])])    
    d_spikerate=np.zeros([len(scanpar[0]), len(scanpar[1])])
    d_spikerate2=np.zeros([len(scanpar[0]), len(scanpar[1])])
    
    # fudgeweights = np.arange(100., 200., 5.)   # Scan for weights fudgefactor
    # d_spikerate=np.zeros([len(fudgeweights),3]) 
    # scanpar=[fudgeweights]
    # d_spikerate=np.zeros(len(scanpar[0]))

    """ GRID HERE """
    # print 'No. of simulations: ', np.size(scanpar)
    for s0 in range(len(scanpar[0])):
        for s1 in range(len(scanpar[1])):
            # for s2 in range(len(scanpar[2])):
            #     for s3 in range(len(scanpar[3])):
            print 'Run # ', k
            modinput={'input_noiserate':  input_noiserates[s0], 'weight_noiseinput': input_weights[s1]}
            
            # Run + readout    
            mo=build_network(modinput)
            nest.Simulate(time_ms) 
            d_spikerate[s0, s1] = float(nest.GetStatus(mo['sd_ca3'],'n_events')[0])/mo['n_ca3e']/time_ms*1000.
            d_spikerate2[s0, s1] = (nest.GetStatus(mo['sd_ca3i'],'n_events')[0]/mo['n_ca3i']/time_ms)*1000.
            print '[CA3 e] Spike rate: ',  float(nest.GetStatus(mo['sd_ca3'],'n_events')[0])/float(mo['n_ca3e'])/time_ms*1000., ' Hz'
            print '[CA3 i] Spike rate: ',  float(nest.GetStatus(mo['sd_ca3i'],'n_events')[0])/float(mo['n_ca3i'])/time_ms*1000., ' Hz'
            print 'Rates: ',  d_spikerate[s0, s1], ' ' ,d_spikerate2[s0, s1]
            k=k+1

    # Return/output of scan
    scanout=locals()
    return scanout

#%% Just run the mother
def JustRun(time_ms):

    mo=build_network(1) # Setup + readout    
    nest.Simulate(time_ms)

    # Print node performance 
    vm= nest.GetStatus(mo['vlt_ca3'],'events')[0] # CA3
    print '[CA3 e] Spike rate: ',  float(nest.GetStatus(mo['sd_ca3'],'n_events')[0])/float(mo['n_ca3e'])/time_ms*1000., ' Hz'
    # print "[CA3 e] Last 10 timesteps Vm: ", vm['V_m'][-10:]
    vm = nest.GetStatus(mo['vlt_ca3i'],'events')[0] 
    print '[CA3 i] Spike rate: ',  float(nest.GetStatus(mo['sd_ca3i'],'n_events')[0])/float(mo['n_ca3i'])/time_ms*1000., ' Hz'
    # print "[CA3 i] Last 10 timesteps Vm: ", vm['V_m'][-10:]
    vm= nest.GetStatus(mo['vlt_ca1'],'events')[0] # CA1
    print '[CA1 e] Spike rate: ',  float(nest.GetStatus(mo['sd_ca1'],'n_events')[0])/float(mo['n_ca1e'])/time_ms*1000., ' Hz'
    # print "[CA1 e] Last 10 timesteps Vm: ", vm['V_m'][-10:]
    vm = nest.GetStatus(mo['vlt_ca1i'],'events')[0] 
    print '[CA1 i] Spike rate: ',  float(nest.GetStatus(mo['sd_ca1i'],'n_events')[0])/float(mo['n_ca1i'])/time_ms*1000., ' Hz'
    # print "[CA1 i] Last 10 timesteps Vm: ", vm['V_m'][-10:]
    vm= nest.GetStatus(mo['vlt_amy'],'events')[0] # CA3
    print '[AMY] Spike rate: ',  float(nest.GetStatus(mo['sd_amy'],'n_events')[0])/float(mo['n_amye'])/time_ms*1000., ' Hz'
    # print "[AMY] Last 10 timesteps Vm: ", vm['V_m'][-10:]
    

    # Spike stats 
    ss=nest.GetStatus(mo['sd_ca3'], 'events')
    spiked_ca3=np.dstack((ss[0]['senders'], ss[0]['times']))
    spiked_ca3.sort(axis=1)  # this is a 2d array, row= event, col 1=neuronid, col 2=spiketime. Get from this the stats divided bt each nruon

    # Return nodes etc as a dictionary
    mo.update({'time_ms':time_ms, 'spiked_ca3':spiked_ca3})
    return mo
    
def plot_spikesvolts(mo):
    fig = plt.figure(figsize=(18,8))
    # fig.set_figheight(20)
    # fig.set_figwidth(80)
    fig_ncols=11
    fig_nrows=4
    fig_plotspikes_nneurons=100
    fig_rasterdot_size=2
    fig_histbinsize=20
    fig_xtick_ms=500
    try:
        fig_xtick_tfbins=int(np.shape(mo['theta_tfpower'])[1])
    except:
        pass

    # plt.xticks(range(0, mo['time_ms'], fig_xtick_ms))

    # Poisson Input
    k=1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('Poisson noise')
    spiketimes=nest.GetStatus(mo['sd_noiseinput'], 'events')[0]['times'] # Spike times 
    neuronidlist=np.ones(len(spiketimes))  
    plt.scatter(spiketimes, neuronidlist, s=fig_rasterdot_size)
    plt.xlim(xmax=mo['time_ms'], xmin=0)
    plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
    k=k+2

    # Theta node
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('Theta spikes')
    try:
        # voltrace=nest.GetStatus(mo['vlt_theta'],'events')[0]['V_m']  # Vm
        # plt.plot(range(len(voltrace)), voltrace)
        spiketimes=nest.GetStatus(mo['sd_theta'], 'events')[0]['times'] # Spike times 
        neuronidlist=np.ones(len(spiketimes))  
        plt.scatter(spiketimes, neuronidlist, s=fig_rasterdot_size*5)
        plt.xlim(xmax=mo['time_ms'], xmin=0)            
        plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
        k=k+1
        plt.subplot(fig_nrows, fig_ncols, k)  # Power analysis
        plt.title('[ThetaIn] FreqPower')    
        plt.plot(mo['theta_power'])
        plt.xlim(xmax=mo['time_ms']/2, xmin=0)
        plt.xticks(range(0, int(mo['time_ms'])/2+1, fig_xtick_ms))
        k=k+1
        plt.subplot(fig_nrows, fig_ncols, k)  # TimexFreq power spectogram
        plt.title('[ThetaIn] TimeFreq Spect')    
        plt.imshow(mo['theta_tfpower'])
        plt.xlim(xmax=fig_xtick_tfbins, xmin=0)
    except:
        pass

    # CA3
    k=fig_ncols+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('[CA3 e] Vm')
    voltrace=nest.GetStatus(mo['vlt_ca3'],'events')[0]['V_m']  # Vm
    plt.plot(range(len(voltrace)), voltrace)
    plt.ylim(ymax=-50, ymin=-75)
    plt.xlim(xmax=mo['time_ms'], xmin=0)
    plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
    k=k+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('Spikes')
    spiketimes=nest.GetStatus(mo['sd_ca3'], 'events')[0]['times'] # Spike times 
    neuronidlist=nest.GetStatus(mo['sd_ca3'], 'events')[0]['senders'] 
    plt.scatter(spiketimes, neuronidlist, s=fig_rasterdot_size)
    try:
        plt.xlim(xmax=mo['time_ms'], xmin=0)
        plt.ylim(ymin= np.unique(neuronidlist)[0],  ymax=np.unique(neuronidlist)[0]+fig_plotspikes_nneurons)
        plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
        k=k+1
        plt.subplot(fig_nrows, fig_ncols, k)  # Spike histogram
        plt.title('Spike hist')    
        plt.hist(spiketimes, bins = range(int(np.min(spiketimes)),  int(np.max(spiketimes)), fig_histbinsize))
        plt.xlim(xmax=mo['time_ms'], xmin=0)
        plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
        k=k+1
        plt.subplot(fig_nrows, fig_ncols, k)  # Power analysis
        plt.title('FreqPower')    
        plt.plot(mo['ca3_power'])
        plt.xlim(xmax=mo['time_ms']/2, xmin=0)
        plt.xticks(range(0, int(mo['time_ms'])/2+1, fig_xtick_ms))
        k=k+1
        plt.subplot(fig_nrows, fig_ncols, k)  # TimexFreq power spectogram
        plt.title('Time x Freq spect')    
        plt.imshow(mo['ca3_tfpower'])
        plt.xlim(xmax=int(np.shape(mo['ca3_tfpower'])[1]), xmin=0)
    except:
        pass
        k=k+3
    k=k+1+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('[CA3 i] Vm')
    voltrace=nest.GetStatus(mo['vlt_ca3i'],'events')[0]['V_m']  # Vm
    plt.plot(range(len(voltrace)), voltrace)
    plt.ylim(ymax=-50, ymin=-75)
    plt.xlim(xmax=mo['time_ms'], xmin=0)
    plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
    k=k+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('Spikes')
    spiketimes=nest.GetStatus(mo['sd_ca3i'], 'events')[0]['times'] # Spike times 
    neuronidlist=nest.GetStatus(mo['sd_ca3i'], 'events')[0]['senders'] 
    plt.scatter(spiketimes, neuronidlist, s=fig_rasterdot_size)
    try:
        plt.xlim(xmax=mo['time_ms'], xmin=0)
        plt.ylim(ymin= np.unique(neuronidlist)[0],  ymax=np.unique(neuronidlist)[0]+fig_plotspikes_nneurons)
        plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
        k=k+1
        plt.subplot(fig_nrows, fig_ncols, k)  # Spike histogram
        plt.title('Spike hist')    
        plt.hist(spiketimes, bins = range(int(np.min(spiketimes)),  int(np.max(spiketimes)), fig_histbinsize))
        plt.xlim(xmax=mo['time_ms'], xmin=0)
        plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
        k=k+1
        plt.subplot(fig_nrows, fig_ncols, k)  # Power analysis
        plt.title('FreqPower')    
        plt.plot(mo['ca3i_power'])
        plt.xlim(xmax=mo['time_ms']/2, xmin=0)
        plt.xticks(range(0, int(mo['time_ms'])/2+1, fig_xtick_ms))
        k=k+1
        plt.subplot(fig_nrows, fig_ncols, k)  # TimexFreq power spectogram
        plt.title('Time x Freq spect')    
        plt.imshow(mo['ca3i_tfpower'])
        plt.xlim(xmax=int(np.shape(mo['ca3i_tfpower'])[1]), xmin=0)
    except:
        pass
        k=k+3
    k=k+1

    # CA1  
    k=2*fig_ncols+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('[CA1 e] Vm')
    voltrace=nest.GetStatus(mo['vlt_ca1'],'events')[0]['V_m']  # Vm
    plt.plot(range(len(voltrace)), voltrace)
    plt.ylim(ymax=-50, ymin=-75)
    plt.xlim(xmax=mo['time_ms'], xmin=0)
    plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
    k=k+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('Spikes')
    spiketimes=nest.GetStatus(mo['sd_ca1'], 'events')[0]['times'] # Spike times 
    neuronidlist=nest.GetStatus(mo['sd_ca1'], 'events')[0]['senders'] 
    plt.scatter(spiketimes, neuronidlist, s=fig_rasterdot_size)
    try:
        plt.xlim(xmax=mo['time_ms'], xmin=0)
        plt.ylim(ymin= np.unique(neuronidlist)[0],  ymax=np.unique(neuronidlist)[0]+fig_plotspikes_nneurons)
        plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
        k=k+1
        plt.subplot(fig_nrows, fig_ncols, k)  # Spike histogram
        plt.title('Spike hist')    
        plt.hist(spiketimes, bins = range(int(np.min(spiketimes)),  int(np.max(spiketimes)), fig_histbinsize))
        plt.xlim(xmax=mo['time_ms'], xmin=0)
        plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
        k=k+1
        plt.subplot(fig_nrows, fig_ncols, k)  # Power analysis
        plt.title('FreqPower')    
        plt.plot(mo['ca1_power'])
        plt.xlim(xmax=mo['time_ms']/2, xmin=0)
        plt.xticks(range(0, int(mo['time_ms'])/2+1, fig_xtick_ms))
        k=k+1
        plt.subplot(fig_nrows, fig_ncols, k)  # TimexFreq power spectogram
        plt.title('Time x Freq spect')    
        plt.imshow(mo['ca1_tfpower'])
        plt.xlim(xmax=int(np.shape(mo['ca1_tfpower'])[1]), xmin=0)
    except:
        pass
        k=k+3
    k=k+1+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('[CA1 i] Vm')
    voltrace=nest.GetStatus(mo['vlt_ca1i'],'events')[0]['V_m']  # Vm
    plt.plot(range(len(voltrace)), voltrace)
    plt.ylim(ymax=-50, ymin=-75)
    plt.xlim(xmax=mo['time_ms'], xmin=0)
    plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
    k=k+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('Spikes')
    spiketimes=nest.GetStatus(mo['sd_ca1i'], 'events')[0]['times'] # Spike times 
    neuronidlist=nest.GetStatus(mo['sd_ca1i'], 'events')[0]['senders'] 
    plt.scatter(spiketimes, neuronidlist, s=fig_rasterdot_size)
    try:
        plt.xlim(xmax=mo['time_ms'], xmin=0)
        plt.ylim(ymin= np.unique(neuronidlist)[0],  ymax=np.unique(neuronidlist)[0]+fig_plotspikes_nneurons)
        plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
        k=k+1
        plt.subplot(fig_nrows, fig_ncols, k)  
        plt.title('Spike hist')    
        plt.hist(spiketimes, bins = range(int(np.min(spiketimes)),  int(np.max(spiketimes)), fig_histbinsize))
        plt.xlim(xmax=mo['time_ms'], xmin=0)
        plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
        k=k+1
        plt.subplot(fig_nrows, fig_ncols, k)  # Power analysis
        plt.title('FreqPower')    
        plt.plot(mo['ca1_power'])
        plt.xlim(xmax=mo['time_ms']/2, xmin=0)
        plt.xticks(range(0, int(mo['time_ms'])/2+1, fig_xtick_ms))
        k=k+1
        plt.subplot(fig_nrows, fig_ncols, k)  # TimexFreq power spectogram
        plt.title('Time x Freq spect')    
        plt.imshow(mo['ca1i_tfpower'])
        plt.xlim(xmax=int(np.shape(mo['ca1i_tfpower'])[1]), xmin=0)          
    except:
        pass
        k=k+3
    k=k+1

    # AMY
    k=3*fig_ncols+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('[AMY] Vm')
    voltrace=nest.GetStatus(mo['vlt_amy'],'events')[0]['V_m']  # Vm
    plt.plot(range(len(voltrace)), voltrace)
    plt.ylim(ymax=-50, ymin=-75)
    plt.xlim(xmax=mo['time_ms'], xmin=0)
    plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
    k=k+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('Spikes')
    spiketimes=nest.GetStatus(mo['sd_amy'], 'events')[0]['times'] # Spike times 
    neuronidlist=nest.GetStatus(mo['sd_amy'], 'events')[0]['senders'] 
    plt.scatter(spiketimes, neuronidlist, s=fig_rasterdot_size)
    try:
        plt.xlim(xmax=mo['time_ms'], xmin=0)
        plt.ylim(ymin= np.unique(neuronidlist)[0],  ymax=np.unique(neuronidlist)[0]+fig_plotspikes_nneurons)
        plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
        k=k+1
        plt.subplot(fig_nrows, fig_ncols, k)  
        plt.title('Spike hist')    
        plt.hist(spiketimes, bins = range(int(np.min(spiketimes)),  int(np.max(spiketimes)), fig_histbinsize))
        plt.xlim(xmax=mo['time_ms'], xmin=0)
        plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
        k=k+1
        plt.subplot(fig_nrows, fig_ncols, k)  # Power analysis
        plt.title('FreqPower')    
        plt.plot(mo['amy_power'])
        plt.xlim(xmax=mo['time_ms']/2, xmin=0)
        plt.xticks(range(0, int(mo['time_ms'])/2+1, fig_xtick_ms))
        k=k+1
        plt.subplot(fig_nrows, fig_ncols, k)  # TimexFreq power spectogram
        plt.title('Time x Freq spect')    
        plt.imshow(mo['amy_tfpower'])       
        plt.xlim(xmax=int(np.shape(mo['amy_tfpower'])[1]), xmin=0)
    except:
        pass
    k=k+1

    # Display !! 
    plt.subplots_adjust(left=0.05, bottom=None, right=0.99, top=None, wspace=0.7, hspace=0.5)
    plt.show()



def spikes2powerspec_alltime(mo, nodename):

    # Get spike data in raster (row=neuron, col=timebin)
    exec("spikedetector=mo['sd_" + nodename + "']")
    exec "n_neurons=len(nest.GetConnections(target=mo['sd_"+nodename + "']))"
    ss=nest.GetStatus(spikedetector, 'events')
    spiket=np.round(ss[0]['times'])
    spikewho=ss[0]['senders']
    spikeraster=np.zeros([n_neurons, int(mo['time_ms'])])   # Binary spike/not for each neuron x timestep. row = neuron, col = time bin (1 ms)
    for e in range(len(ss[0]['times'])):
        spikeraster[spikewho[e]-np.min(spikewho)-1, spiket[e]-1]=1

    # Autocorrelation
    autocor_spiketrain=np.zeros([n_neurons, int(mo['time_ms'])]) # Autocorrelation plots (row=neuron)
    for n in range(n_neurons):
        ast=np.correlate(spikeraster[n,:],spikeraster[n,:], mode='full')
        autocor_spiketrain[n,:]=ast[ast.size/2:]

    # Fourier
    # Simple frequency spectrum [http://pythonhosted.org/NeuroTools/_modules/NeuroTools/analysis.html#simple_frequency_spectrum]
    # Very simple calculation of frequency spectrum with no detrending, windowing, etc, just the first half 
    # (positive frequency components) of abs(fft(x))  
    #
    # Parameters
    # ----------
    # x : array_like   -  The input array, in the time-domain.
    #
    # Returns
    # -------
    # spec : array_like
    #     The frequency spectrum of `x`.
    powerspex=np.zeros([n_neurons, int(mo['time_ms'])/2]) # Power spectrum data (row=neuron)
    for n in range(n_neurons):
        x=autocor_spiketrain[n,:]
        spec = np.absolute(np.fft.fft(x))
        spec = spec[:len(x) / 2]  # take positive frequency components
        spec /= len(x)  # normalize (by timesteps)
        spec *= 2.0  # to get amplitudes of sine components, need to multiply by 2
        spec[0] /= 2.0  # except for the dc component
        powerspex[n,:]=spec
    poppower=powerspex.mean(axis=0)
    return spikeraster, autocor_spiketrain, powerspex, poppower



# def spikes2powerspec_timebins(mo, nodename, timebin_ms):

#     # Get spike data in raster (row=neuron, col=timebin)
#     n_bins=int(mo['time_ms']/timebin_ms)
#     n_autocorss=n_bins+1 # Assuming 50% bin overlap
#     exec("spikedetector=mo['sd_" + nodename + "']")
#     exec "n_neurons=len(nest.GetConnections(target=mo['sd_"+nodename + "']))"
#     ss=nest.GetStatus(spikedetector, 'events')
#     spiket=np.round(ss[0]['times'])
#     spikewho=ss[0]['senders']
#     spikeraster=np.zeros([n_neurons, int(mo['time_ms'])])   # Binary spike/not for each neuron x timestep. row = neuron, col = time bin (1 ms)
#     for e in range(len(ss[0]['times'])):
#         spikeraster[spikewho[e]-np.min(spikewho)-1, spiket[e]-1]=1

#     # Autocorrelation within time bins (50% overlap)
#     autocor_binstart=np.unique(np.round(np.arange(0, int(mo['time_ms']), int(timebin_ms)/2))) # Indexing the spikes stored in spikeraster
#     print autocor_binstart
#     powerspec=np.zeros((int(timebin_ms/2), len(autocor_binstart)-1))   # x/col=time, y/row=freq
#     for bs in range(len(autocor_binstart)-1):  # For all (timebinned) snips
#         freq_thissnipbin=np.zeros((int(timebin_ms/2)))
#         for n in range(n_neurons):  # For all neurons
#             thisneuronsnip=spikeraster[n, autocor_binstart[bs]:(autocor_binstart[bs]+int(timebin_ms))]
#             if not thisneuronsnip.size==int(timebin_ms):
#                 print 'Check snip size (sliding window power analysis)'
#             thisneuronsnip_autocor=np.correlate(thisneuronsnip, thisneuronsnip, mode='full') # Autocorrelate
#             thisneuronsnip_autocor=thisneuronsnip_autocor[thisneuronsnip_autocor.size/2:]
#             x=thisneuronsnip_autocor  # Fourier
#             spec = np.absolute(np.fft.fft(x))
#             spec = spec[:len(x) / 2]  # take positive frequency components
#             spec /= len(x)  # normalize (by timesteps)
#             spec *= 2.0  # to get amplitudes of sine components, need to multiply by 2
#             spec[0] /= 2.0  # except for the dc component
#             freq_thissnipbin=freq_thissnipbin + spec
#         freq_thissnipbin=freq_thissnipbin/n_neurons  # Mean freq vector of all neurons
#         powerspec[:, bs]=freq_thissnipbin # Record to overall time x freq plot. y/row=freq, x/col=timebin
#     return  spikeraster, powerspec


"##########################################################################################"
"##########################################################################################"
"######################## WHAT DO YOU WANT TO DO RIGHT NOW ? ##############################"
"##########################################################################################"
"##########################################################################################"


""" [1] Parameter search  """
# so=scan_weights(10000.)  # Parameter search
# print so['d_spikerate']
# print so['scanpar']
# plot_spikesvolts(so['mo'])

""" [2] Just run the model + power analysis """
mo=JustRun(1000.)
# spikeraster, autocor_spiketrain, powerspex, mo['theta_power']=spikes2powerspec_alltime(mo, 'theta')  # Frequency analysis
# spikeraster, autocor_spiketrain, powerspex, mo['ca3_power']=spikes2powerspec_alltime(mo, 'ca3')  
# spikeraster, autocor_spiketrain, powerspex, mo['ca3i_power']=spikes2powerspec_alltime(mo, 'ca3i')
# spikeraster, autocor_spiketrain, powerspex, mo['ca1_power']=spikes2powerspec_alltime(mo, 'ca1')
# spikeraster, autocor_spiketrain, powerspex, mo['ca1i_power']=spikes2powerspec_alltime(mo, 'ca1i')
# spikeraster, autocor_spiketrain, powerspex, mo['amy_power']=spikes2powerspec_alltime(mo, 'amy')
# binsize_ms=50.      # Sliding window power spectrogram
# mo['binsize_ms']=binsize_ms
# spikeraster, mo['theta_tfpower']=spikes2powerspec_timebins(mo, 'theta', binsize_ms)
# spikeraster, mo['ca3_tfpower']=spikes2powerspec_timebins(mo, 'ca3', binsize_ms)
# spikeraster, mo['ca3i_tfpower']=spikes2powerspec_timebins(mo, 'ca3i', binsize_ms)
# spikeraster, mo['ca1_tfpower']=spikes2powerspec_timebins(mo, 'ca1', binsize_ms)
# spikeraster, mo['ca1i_tfpower']=spikes2powerspec_timebins(mo, 'ca1i', binsize_ms)
# spikeraster, mo['amy_tfpower']=spikes2powerspec_timebins(mo, 'amy', binsize_ms)
# plot_spikesvolts(mo)



# Test plot heatmap
# plt.imshow(mo['ca3_tfpower'])
# plt.colorbar
# plt.show()




""" Sliding window freq analysis """
# def spikes2powerspec_timebins(mo, nodename, timebin_ms):
nodename='ca3'
timebin_ms= 50.

# Get spike data in raster (row=neuron, col=timebin)
# n_bins=int(mo['time_ms']/timebin_ms)
# n_autocorss=n_bins+1 # Assuming 50% bin overlap
# exec("spikedetector=mo['sd_" + nodename + "']")
# exec "n_neurons=len(nest.GetConnections(target=mo['sd_"+nodename + "']))"
# ss=nest.GetStatus(spikedetector, 'events')
# spiket=np.round(ss[0]['times'])
# spikewho=ss[0]['senders']
# spikeraster=np.zeros([n_neurons, int(mo['time_ms'])])   # Binary spike/not for each neuron x timestep. row = neuron, col = time bin (1 ms)
# for e in range(len(ss[0]['times'])):
#     spikeraster[spikewho[e]-np.min(spikewho)-1, spiket[e]-1]=1

# # Autocorrelation within time bins (50% overlap)
# autocor_binstart=np.unique(np.round(np.arange(0, int(mo['time_ms']), int(timebin_ms)/2))) # Indexing the spikes stored in spikeraster
# print autocor_binstart
# powerspec=np.zeros((int(timebin_ms/2), len(autocor_binstart)-1))   # x/col=time, y/row=freq
# for bs in range(len(autocor_binstart)-1):  # For all (timebinned) snips
#     freq_thissnipbin=np.zeros((int(timebin_ms/2)))
#     for n in range(n_neurons):  # For all neurons
#         thisneuronsnip=spikeraster[n, autocor_binstart[bs]:(autocor_binstart[bs]+int(timebin_ms))]
#         if not thisneuronsnip.size==int(timebin_ms):
#             print 'Check snip size (sliding window power analysis)'
#         thisneuronsnip_autocor=np.correlate(thisneuronsnip, thisneuronsnip, mode='full') # Autocorrelate
#         thisneuronsnip_autocor=thisneuronsnip_autocor[thisneuronsnip_autocor.size/2:]
#         x=thisneuronsnip_autocor  # Fourier
#         spec = np.absolute(np.fft.fft(x))
#         spec = spec[:len(x) / 2]  # take positive frequency components
#         spec /= len(x)  # normalize (by timesteps)
#         spec *= 2.0  # to get amplitudes of sine components, need to multiply by 2
#         spec[0] /= 2.0  # except for the dc component
#         freq_thissnipbin=freq_thissnipbin + spec
#     freq_thissnipbin=freq_thissnipbin/n_neurons  # Mean freq vector of all neurons
#     powerspec[:, bs]=freq_thissnipbin # Record to overall time x freq plot. y/row=freq, x/col=timebin
# # return  spikeraster, powerspec



    