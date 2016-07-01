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
    input_noiserate=19000.      # input rate & weight= 19000 & 9 will get CA3 down to <8 Hz (excit and inhib oscillate)
    weight_noiseinput=9.
    # input_noiserate=20400.      # input rate & weight= 20500 & 8.25 will get CA3 down to <10 Hz (excit and inhib oscillate)
    # weight_noiseinput=8.25
    # input_noiserate=20550.      # input rate & weights arbitrary
    # weight_noiseinput=8.25
    c_inhib_weightmultiplier= 90./71.
    c_fudgeweights=10.5
    # c_ca3_2inhib_bonus=3.
    c_ca3_2inhib_bonus= 3.
    c_ca3_recur_bonus= -0.2



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
    # ns_ca1e_mempot=ns_ca3e_mempot   # FAKE CA1 and Amygdala
    # n_ca1e=n_ca3e   
    # n_ca1i=n_ca3i

    # Connectivity = [ConnectivityProbability  Weights]  Intrinsic connex from Taxidis 2011 HPC
    c_ca3_recur= [0.1062,  (1.+c_ca3_recur_bonus)*c_fudgeweights]    # Intra-CA3
    c_ca3_2inhib= [0.0988,  (1.6+c_ca3_2inhib_bonus)*c_fudgeweights]
    c_ca3_inhib2excit= [0.713, 1.2*c_fudgeweights* -1 * c_inhib_weightmultiplier]
    c_ca1_2inhib= [0.3111,  1.3*c_fudgeweights]    # Intra-CA1
    c_ca1_inhib2excit=[0.8397,  0.5*c_fudgeweights*-1 * c_inhib_weightmultiplier]
    c_ca1_inhib2inhib=[0.9089,  0.5*c_fudgeweights*-1 * c_inhib_weightmultiplier]
    c_ca3_2ca1= [0.1821,  0.13*c_fudgeweights]  # CA3 --> CA1 (Schaffer)
    c_ca3_2ca1i= [0.1825,  0.4*c_fudgeweights]  # CA3 --> CA1 interneurons
    # Weights from here are arbitrary !!! 
    c_ca3_2amye=[0.35,  2.*c_fudgeweights]
    c_amye_2ca1e=[0.8,  1.5*c_fudgeweights]


    # # Dynamics
    theta_amp=7.
    theta_freq= 7.5  # theta freq: 6-10 Hz
    theta_offset=0.
    theta_phase=0.
    weight_thetainput=0.4
    
    # ################################################################################
    # Create nodes + Readouts
    nest.CopyModel("static_synapse","readout", {"weight":1., "delay": dt})  
    input_noise=nest.Create("poisson_generator", 1, {'rate': input_noiserate, 'start': 0., } )
    sd_noiseinput=nest.Create("spike_detector") # , params={"to_file": True, "withtime": True, "label":'sd_ca3'})  
    nest.Connect(input_noise, sd_noiseinput, model='readout')
    input_theta=nest.Create("ac_generator", 1, {'amplitude': theta_amp, 'offset':theta_offset,  'phase': theta_phase,  'frequency': theta_freq} )
    vlt_theta= nest.Create("multimeter") # , params={"to_file": True, "withtime": True,"label":'vlt_theta'})  
    nest.Connect(vlt_theta, input_theta, model='readout')
    # input_noise=nest.Create("poisson_generator", 1, {'rate': input_noiserate, 'start': 0., } )
    # sd_noiseinput=nest.Create("spike_detector") # , params={"to_file": True, "withtime": True, "label":'sd_ca3'})  
    # nest.Connect(input_noise, sd_noiseinput, model='readout')
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



    """ Fake weights """
    # c_ca3_recur= [0.1062 * 0.5,  1.*c_fudgeweights]    # Intra-CA3
    # c_ca3_2inhib= [0.0988,  1.6*c_fudgeweights + c_ca3_2inhib_bonus*c_fudgeweights]
    # c_ca3_inhib2excit= [0.713, 1.2*c_fudgeweights* -1 * c_inhib_weightmultiplier]
    # c_ca1_2inhib= [0.3111,  1.3*c_fudgeweights]    # Intra-CA1
    # c_ca1_inhib2excit=[0.8397,  0.5*c_fudgeweights*-1 * c_inhib_weightmultiplier]
    # c_ca1_inhib2inhib=[0.9089,  0.5*c_fudgeweights*-1 * c_inhib_weightmultiplier]
    # c_ca3_2ca1= [0.1821,  0.13*c_fudgeweights]  # CA3 --> CA1 (Schaffer)
    # c_ca3_2ca1i= [0.1825,  0.4*c_fudgeweights]  # CA3 --> CA1 interneurons
    # # Weights from here are arbitrary !!! 
    # c_ca3_2amye=[0.35,  2.*c_fudgeweights]
    # c_amye_2ca1e=[0.8,  1.5*c_fudgeweights]





    """ Connectivity """
    nest.DivergentConnect(input_noise, pope_ca3, weight_noiseinput, dt)  
    nest.DivergentConnect(input_theta, pope_ca3, weight_thetainput, dt)  

    # vlt_theta
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
    
    # To amygdala
    nest.RandomDivergentConnect(pope_ca3, pope_amy, int(round(c_ca3_2amye[0]*n_amye)), weight=c_ca3_2amye[1], delay=1.)   
    nest.RandomDivergentConnect(pope_amy, pope_ca1, int(round(c_amye_2ca1e[0]*n_ca1e)), weight=c_amye_2ca1e[1], delay=1.)   


    """ Fake connectivity things """
    nest.DivergentConnect(input_noise, popi_ca3, weight_noiseinput*0.1, dt)  




    # ################################################################################
    modoutput=locals()
    return modoutput

def scan_weights():
    """ Scan across params """
    time_ms=1000.
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
def JustRun():

    mo=build_network(1) # Setup + readout    
    time_ms=1000.
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
    fig_ncols=4+2+1
    fig_nrows=4
    fig_plotspikes_nneurons=100
    fig_histbinsize=50
    fig_xtick_ms=500

    # plt.xticks(range(0, mo['time_ms'], fig_xtick_ms))

    # Poisson Input
    k=1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('Poisson/Noise input')
    spiketimes=nest.GetStatus(mo['sd_noiseinput'], 'events')[0]['times'] # Spike times 
    neuronidlist=np.ones(len(spiketimes))  
    plt.scatter(spiketimes, neuronidlist)
    plt.xlim(xmax=mo['time_ms'], xmin=0)
    plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
    k=k+1

    # Theta node
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('Theta ac')
    try:
        voltrace=nest.GetStatus(mo['vlt_theta'],'events')[0]['V_m']  # Vm
        plt.plot(range(len(voltrace)), voltrace)
        plt.xlim(xmax=mo['time_ms'], xmin=0)            
        plt.ylim(ymax=-50, ymin=-75)
        plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
    except:
        pass

    # CA3
    k=fig_ncols+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('[CA3 e] voltage')
    voltrace=nest.GetStatus(mo['vlt_ca3'],'events')[0]['V_m']  # Vm
    plt.plot(range(len(voltrace)), voltrace)
    plt.ylim(ymax=-50, ymin=-75)
    plt.xlim(xmax=mo['time_ms'], xmin=0)
    plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
    k=k+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('[CA3 e] spikes')
    spiketimes=nest.GetStatus(mo['sd_ca3'], 'events')[0]['times'] # Spike times 
    neuronidlist=nest.GetStatus(mo['sd_ca3'], 'events')[0]['senders'] 
    plt.scatter(spiketimes, neuronidlist)
    try:
        plt.xlim(xmax=mo['time_ms'], xmin=0)
        plt.ylim(ymin= np.unique(neuronidlist)[0],  ymax=np.unique(neuronidlist)[0]+fig_plotspikes_nneurons)
        plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
        k=k+1
        plt.subplot(fig_nrows, fig_ncols, k)  
        plt.title('[CA3 e] Spike hist')    
        plt.hist(spiketimes, bins = range(int(np.min(spiketimes)),  int(np.max(spiketimes)), fig_histbinsize))
        plt.xlim(xmax=mo['time_ms'], xmin=0)
        plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
    except:
        pass
        k=k+1
    k=k+1+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('[CA3 i] voltage')
    voltrace=nest.GetStatus(mo['vlt_ca3i'],'events')[0]['V_m']  # Vm
    plt.plot(range(len(voltrace)), voltrace)
    plt.ylim(ymax=-50, ymin=-75)
    plt.xlim(xmax=mo['time_ms'], xmin=0)
    plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
    k=k+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('[CA3 i] spikes')
    spiketimes=nest.GetStatus(mo['sd_ca3i'], 'events')[0]['times'] # Spike times 
    neuronidlist=nest.GetStatus(mo['sd_ca3i'], 'events')[0]['senders'] 
    plt.scatter(spiketimes, neuronidlist)
    try:
        plt.xlim(xmax=mo['time_ms'], xmin=0)
        plt.ylim(ymin= np.unique(neuronidlist)[0],  ymax=np.unique(neuronidlist)[0]+fig_plotspikes_nneurons)
        plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
        k=k+1
        plt.subplot(fig_nrows, fig_ncols, k)  
        plt.title('[CA3 i] Spike hist')    
        plt.hist(spiketimes, bins = range(int(np.min(spiketimes)),  int(np.max(spiketimes)), fig_histbinsize))
        plt.xlim(xmax=mo['time_ms'], xmin=0)
        plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
    except:
        pass
        k=k+1
    k=k+1

    # CA1  
    k=2*fig_ncols+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('[CA1 e] voltage')
    voltrace=nest.GetStatus(mo['vlt_ca1'],'events')[0]['V_m']  # Vm
    plt.plot(range(len(voltrace)), voltrace)
    plt.ylim(ymax=-50, ymin=-75)
    plt.xlim(xmax=mo['time_ms'], xmin=0)
    plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
    k=k+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('[CA1 e] spikes')
    spiketimes=nest.GetStatus(mo['sd_ca1'], 'events')[0]['times'] # Spike times 
    neuronidlist=nest.GetStatus(mo['sd_ca1'], 'events')[0]['senders'] 
    plt.scatter(spiketimes, neuronidlist)
    try:
        plt.xlim(xmax=mo['time_ms'], xmin=0)
        plt.ylim(ymin= np.unique(neuronidlist)[0],  ymax=np.unique(neuronidlist)[0]+fig_plotspikes_nneurons)
        plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
        k=k+1
        plt.subplot(fig_nrows, fig_ncols, k)  
        plt.title('[CA1 e] Spike hist')    
        plt.hist(spiketimes, bins = range(int(np.min(spiketimes)),  int(np.max(spiketimes)), fig_histbinsize))
        plt.xlim(xmax=mo['time_ms'], xmin=0)
        plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
    except:
        pass
        k=k+1
    k=k+1+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('[CA1 i] voltage')
    voltrace=nest.GetStatus(mo['vlt_ca1i'],'events')[0]['V_m']  # Vm
    plt.plot(range(len(voltrace)), voltrace)
    plt.ylim(ymax=-50, ymin=-75)
    plt.xlim(xmax=mo['time_ms'], xmin=0)
    plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
    k=k+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('[CA1 i] spikes')
    spiketimes=nest.GetStatus(mo['sd_ca1i'], 'events')[0]['times'] # Spike times 
    neuronidlist=nest.GetStatus(mo['sd_ca1i'], 'events')[0]['senders'] 
    plt.scatter(spiketimes, neuronidlist)
    try:
        plt.xlim(xmax=mo['time_ms'], xmin=0)
        plt.ylim(ymin= np.unique(neuronidlist)[0],  ymax=np.unique(neuronidlist)[0]+fig_plotspikes_nneurons)
        plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
        k=k+1
        plt.subplot(fig_nrows, fig_ncols, k)  
        plt.title('[CA1 i] Spike hist')    
        plt.hist(spiketimes, bins = range(int(np.min(spiketimes)),  int(np.max(spiketimes)), fig_histbinsize))
        plt.xlim(xmax=mo['time_ms'], xmin=0)
        plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
    except:
        pass
        k=k+1
    k=k+1

    # AMY
    k=3*fig_ncols+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('[AMY] voltage')
    voltrace=nest.GetStatus(mo['vlt_amy'],'events')[0]['V_m']  # Vm
    plt.plot(range(len(voltrace)), voltrace)
    plt.ylim(ymax=-50, ymin=-75)
    plt.xlim(xmax=mo['time_ms'], xmin=0)
    plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
    k=k+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('[AMY] spikes')
    spiketimes=nest.GetStatus(mo['sd_amy'], 'events')[0]['times'] # Spike times 
    neuronidlist=nest.GetStatus(mo['sd_amy'], 'events')[0]['senders'] 
    plt.scatter(spiketimes, neuronidlist)
    try:
        plt.xlim(xmax=mo['time_ms'], xmin=0)
        plt.ylim(ymin= np.unique(neuronidlist)[0],  ymax=np.unique(neuronidlist)[0]+fig_plotspikes_nneurons)
        plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
        k=k+1
        plt.subplot(fig_nrows, fig_ncols, k)  
        plt.title('[AMY] Spike hist')    
        plt.hist(spiketimes, bins = range(int(np.min(spiketimes)),  int(np.max(spiketimes)), fig_histbinsize))
        plt.xlim(xmax=mo['time_ms'], xmin=0)
        plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
    except:
        pass
    k=k+1


    # Display !! 
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    plt.show()


def simple_frequency_spectrum(x):
    """Simple frequency spectrum. [http://pythonhosted.org/NeuroTools/_modules/NeuroTools/analysis.html#simple_frequency_spectrum]
    Very simple calculation of frequency spectrum with no detrending, windowing, etc, just the first half 
    (positive frequency components) of abs(fft(x))  

    Parameters
    ----------
    x : array_like   -  The input array, in the time-domain.

    Returns
    -------
    spec : array_like
        The frequency spectrum of `x`.

    """
    spec = np.absolute(np.fft.fft(x))
    spec = spec[:len(x) / 2]  # take positive frequency components
    spec /= len(x)  # normalize
    spec *= 2.0  # to get amplitudes of sine components, need to multiply by 2
    spec[0] /= 2.0  # except for the dc component
    return spec



##########################################################################################
##########################################################################################
######################## WHAT DO YOU WANT TO DO RIGHT NOW ? ##############################
##########################################################################################
##########################################################################################


""" [1] Parameter search  """
# so=scan_weights()  # Parameter search
# print so['d_spikerate']
# print so['d_spikerate2']
# print so['scanpar']
# mo=so['mo']
# plot_spikesvolts(mo)

""" [2] Just run the model"""
mo=JustRun()
plot_spikesvolts(mo)







# Frequency ????
# ox=simple_frequency_spectrum(strain)



    # # Spike stats 
    # ss=nest.GetStatus(mo['sd_ca3'], 'events')
    # spiked=np.dstack((ss[0]['senders'], ss[0]['times']))
    # spiked.sort(axis=1)  # row= event, col 1=neuronid, col 2=spiketime
    # strain=ss[0]['times']
    # plt.hist(strain, bins = range(int(np.min(strain)),  int(np.max(strain)), fig_histbinsize))



# a = [[1,2,3],[4,5,6],[7,8,9]]
# ?ar = array(a)

# import csv

# fl = open('filename.csv', 'w')



# k=1
# writer = csv.writer(fl)
# # writer.writerow(['label1', 'label2', 'label3']) #if needed
# for values in spiked_ca3:
#     print values
#     # print values[0][1]
#     print k
#     k=k+1

    # writer.writerow(values)  # Original code
    # writer.writerow(values[0][0])
    # writer.writerow(values[0][1])

# fl.close()    







# # # # Nest-inspect a particular node
# nest.raster_plot.from_device(mo['sd_ca3'], hist=True)
# nest.raster_plot.show()
# nest.voltage_trace.from_device(mo['vlt_ca3'])

# # a=nest.GetStatus(mo['sd_ca3'])



# # Get Frequency spectrum
# x=nest.GetStatus(mo['sd_ca3'],'events')[0]['times']
# ox=simple_frequency_spectrum(x)


##########################################################################################
######################################  CODE DUMP ########################################
##########################################################################################
# # Fetching behaviours from a node (beh_*: VoltageTrace, n_spikes, spiketimes)
# beh_ca3=[nest.GetStatus(mo['vlt_ca3'],'events')[0]['V_m'],   nest.GetStatus(mo['sd_ca3'],'n_events')[0],  nest.GetStatus(mo['sd_ca3'], 'events')[0]['times']]

# Interrogate connections
# a= nest.GetConnections(source=mo['pope_ca3'],target=mo['popi_ca3'])



# # Nest-inspect a particular node
# nest.raster_plot.from_device(mo['sd_ca1'],hist=True)
# nest.voltage_trace.from_device(mo['vlt_ca1'])
# nest.raster_plot.show()



# pl.subplot(211)
# pl.plot(t, events['V_m'])
# pl.axis([0, 100, -75, -53])
# pl.ylabel('Membrane potential [mV]')

# pl.subplot(212)
# pl.plot(t, events['g_ex'], t, events['g_in'])
# pl.axis([0, 100, 0, 45])
# pl.xlabel('Time [ms]')
# pl.ylabel('Synaptic conductance [nS]')
# pl.legend(('g_exc', 'g_inh'))


# Multiplying all elements in a list
# a1 = [0,1,2] 
# a1[:] = [x*3 for x in a1]   # or replace a1[:] with a2
# 