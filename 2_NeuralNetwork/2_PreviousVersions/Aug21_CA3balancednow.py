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



def build_network(modinput): # (input_rate, weight_input2ca3):

    # Free parameters ######  
    # input_rate=modinput['input_rate']
    # weight_input2ca3=modinput['weight_input2ca3']
    # c_inhib_weightmultiplier=modinput['c_inhib_weightmultiplier']
    # c_fudgeweights=modinput['c_fudgeweights']
    # c_ca3_2inhib_bonus=modinput['c_ca3_2inhib_bonus']
    input_rate=19800.    # SPECIFIED PARAMETER VALUES (from scan)
    weight_input2ca3=9.
    c_inhib_weightmultiplier= 90./71.
    c_fudgeweights=10.5
    c_ca3_2inhib_bonus=3.


    """ Builds the network """
    nest.ResetKernel()
    nest.SetKernelStatus({"overwrite_files": True, 'data_path': '/Users/EleanorL/Dropbox/SANDISK/12_ACCN/1_Project/2_Simulated_data' , "resolution":dt})

    # Node setup
    n_inputs=1
    prop_pyramidpop_e=0.9
    type_ca3='iaf_psc_exp'  # CA3
    n_ca3=1000
    n_ca1=1000
    n_ca3e=int(round(n_ca3*prop_pyramidpop_e))
    n_ca3i=int(round(n_ca3*(1-prop_pyramidpop_e)))
    type_ca1=type_ca3       # CA1 = Inputs and Inhibition only
    # n_ca1e=int(round(n_ca1*prop_pyramidpop_e))
    # n_ca1i=int(round(n_ca1*(1-prop_pyramidpop_e)))
    n_ca1e=n_ca3e   
    n_ca1i=n_ca3i
    ns_ca3e_mempot= - 65.3
    # ns_ca1e_mempot= - 62.6
    ns_ca1e_mempot=ns_ca3e_mempot
    ns_ca_inhib_mempot= - 62.
    n_amye=n_ca3e       # Amygdala: Recurrents only
    type_amy=type_ca3
    ns_amye_mempot=ns_ca3e_mempot

    # Connectivity = [ConnectivityProbability  Weights]  Intrinsic connex from Taxidis 2011 HPC
    c_ca3_recur= [0.1062,  1.*c_fudgeweights]    # Intra-CA3
    c_ca3_2inhib= [0.0988,  1.6*c_fudgeweights + c_ca3_2inhib_bonus*c_fudgeweights]
    c_ca3_inhib2excit= [0.713, 1.2*c_fudgeweights* -1 * c_inhib_weightmultiplier]
    # c_ca1_2inhib= [0.3111,  1.3*c_fudgeweights]    # Intra-CA1
    # c_ca1_inhib2excit=[8397,  0.5*c_fudgeweights*-1 * c_inhib_weightmultiplier]
    # c_ca1_inhib2inhib=[0.9089,  0.5*c_fudgeweights*-1 * c_inhib_weightmultiplier]
    # c_ca3_2ca1= [0.1821,  0.13*c_fudgeweights]  # CA3 --> CA1 (Schaffer)

    """ FAKE WEIGHTS """

    # # Dynamics
    # theta_amp=50.
    # theta_freq= 7.5  # theta freq: 6-10 Hz

    
    # ################################################################################
    # Create nodes + Readouts
    nest.CopyModel("static_synapse","readout", {"weight":1., "delay": dt})  
    input_ca3=nest.Create("poisson_generator", n_inputs, {'rate': input_rate, 'start': 0., } )
    sd_input=nest.Create("spike_detector") # , params={"to_file": True, "withtime": True, "label":'sd_ca3'})  
    nest.Connect(input_ca3, sd_input, model='readout')
    pope_ca3=nest.Create(type_ca3, n_ca3e)  # CA3 ------------------------------
    popi_ca3=nest.Create(type_ca3, n_ca3i)
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
    pope_ca1=nest.Create(type_ca1, n_ca1e)   # CA1 ------------------------------
    popi_ca1=nest.Create(type_ca1, n_ca1i)
    noise = np.random.normal(nest.GetStatus(pope_ca1)[2]['V_m'], 2, len(pope_ca1))
    for n in range(len(pope_ca1)):
        nest.SetStatus([pope_ca1[n]], {'V_m':noise[n]})
    sd_ca1=nest.Create("spike_detector") # , params={"to_file": True, "withtime": True, "label":'sd_ca1'})  
    vlt_ca1= nest.Create("voltmeter") # , params={"to_file": True, "withtime": True,"label":'vlt_ca1'})  
    sd_ca1i=nest.Create("spike_detector") # , params={"to_file": True, "withtime": True, "label":'sd_ca1i'})  
    vlt_ca1i= nest.Create("voltmeter") # , params={"to_file": True, "withtime": True,"label":'vlt_ca1i'})  
    nest.Connect(vlt_ca1, [pope_ca1[1]], model='readout') 
    nest.ConvergentConnect(pope_ca1, sd_ca1, model='readout')
    nest.Connect(vlt_ca1i, [popi_ca1[1]], model='readout')
    nest.ConvergentConnect(popi_ca1, sd_ca1i, model='readout')
    pope_amy=nest.Create(type_amy, n_amye)   # amy ------------------------------
    # popi_amy=nest.Create(type_amy, n_amyi, {'E_L': ns_ca_inhib_mempot})
    noise = np.random.normal(nest.GetStatus(pope_amy)[2]['V_m'], 2, len(pope_amy))
    for n in range(len(pope_amy)):
        nest.SetStatus([pope_amy[n]], {'V_m':noise[n]})
    sd_amy=nest.Create("spike_detector") # , params={"to_file": True, "withtime": True, "label":'sd_amy'})  
    vlt_amy= nest.Create("voltmeter") # , params={"to_file": True, "withtime": True,"label":'vlt_amy'})  
    sd_amyi=nest.Create("spike_detector") # , params={"to_file": True, "withtime": True, "label":'sd_amyi'})  
    vlt_amyi= nest.Create("voltmeter") # , params={"to_file": True, "withtime": True,"label":'vlt_amyi'})  
    nest.Connect(vlt_amy, [pope_amy[1]], model='readout') 
    nest.ConvergentConnect(pope_amy, sd_amy, model='readout')
    # nest.Connect(vlt_amyi, [popi_amy[1]], model='readout')
    # nest.ConvergentConnect(popi_amy, sd_amyi, model='readout')

    # Connect up nodes ######## 
    nest.DivergentConnect(input_ca3, pope_ca3, weight_input2ca3, dt)  
    nest.RandomDivergentConnect(pope_ca3, pope_ca3, int(round(c_ca3_recur[0]*n_ca3e)), weight=c_ca3_recur[1], delay=1.)   # Within CA3
    nest.RandomDivergentConnect(pope_ca3, popi_ca3, int(round(c_ca3_2inhib[0]*n_ca3i)), weight=c_ca3_2inhib[1], delay=1.)
    nest.RandomDivergentConnect(popi_ca3, pope_ca3, int(round(c_ca3_inhib2excit[0]*n_ca3e)), weight=c_ca3_inhib2excit[1], delay=1.)
    # nest.RandomDivergentConnect(pope_ca3, pope_ca1, int(round(c_ca3_2ca1[0]*n_ca1e)), weight=c_ca3_2ca1[1], delay=1.)   # CA3 --> CA1



    """ Artificially connect up CA1 to the input genetator!!"""
    # CA1= Inhibition only
    nest.DivergentConnect(input_ca3, pope_ca1, weight_input2ca3, dt)  
    # nest.RandomDivergentConnect(pope_ca1, pope_ca1, int(round(c_ca3_recur[0]*n_ca3e)), weight=c_ca3_recur[1], delay=1.)   # Within CA3
    # nest.RandomDivergentConnect(pope_ca1, popi_ca1, int(round(c_ca3_2inhib[0]*n_ca3i)), weight=c_ca3_2inhib[1], delay=1.)
    # nest.RandomDivergentConnect(popi_ca1, pope_ca1, int(round(c_ca3_inhib2excit[0]*n_ca3e)), weight=c_ca3_inhib2excit[1], delay=1.)

    # Amydala = Recurrents only
    nest.DivergentConnect(input_ca3, pope_amy, weight_input2ca3, dt)  
    nest.RandomDivergentConnect(pope_amy, pope_amy, int(round(c_ca3_recur[0]*n_ca3e)), weight=c_ca3_recur[1], delay=1.)   
    

    # ################################################################################
    # Return nodes etc as a dictionary
    # modoutput={'pope_ca3':pope_ca3, 'n_ca3e':n_ca3e, 'sd_ca3':sd_ca3, 'vlt_ca3': vlt_ca3}
    modoutput=locals()
    return modoutput

    
def scan_weights():
    """ Scan across params """
    time_ms=1000.

    # Settings 
    # input_rates = np.arange(19000.,22000., 200.)
    # input_weights =  np.arange(8., 11., 1.)   
    # inhibweight_multiplier=np.arange(0.5, 5., 1.5)   
    # inhibweight_multiplier=[2.]
    fudge_weights=np.arange(0.5, 30., 2.)   
    c_ca3_2inhib_bonus=np.arange(-5., 5., 0.5)   
    scanpar=[c_ca3_2inhib_bonus] #, inhibweight_multiplier, fudge_weights ]
    # print 'No. of simulations: ', len(scanpar[0])*len(scanpar[1])*len(scanpar[2])*len(scanpar[3])
    # d_spikerate=np.zeros([len(scanpar[0]), len(scanpar[1]), len(scanpar[2]), len(scanpar[3])])    
    # d_spikerate2=np.zeros([len(scanpar[0]), len(scanpar[1]), len(scanpar[2]), len(scanpar[3])])    
    d_spikerate=np.zeros([len(scanpar[0]),2])    
    # d_spikerate2=np.zeros([len(scanpar[0])    
    
    # fudgeweights = np.arange(100., 200., 5.)   # Scan for weights fudgefactor
    # d_spikerate=np.zeros([len(fudgeweights),3]) 
    # scanpar=[fudgeweights]
    # d_spikerate=np.zeros(len(scanpar[0]))

    """ GRID HERE """
    # print 'No. of simulations: ', len(scanpar[0])*len(scanpar[1])*len(scanpar[2])*len(scanpar[3])
    for s0 in range(len(scanpar[0])):
        # for s1 in range(len(scanpar[1])):
            # for s2 in range(len(scanpar[2])):
            #     for s3 in range(len(scanpar[3])):
        print s0, # ' - ', s1, # ' - ', s2, ' - ', s3
        modinput={'c_ca3_2inhib_bonus': c_ca3_2inhib_bonus[s0]}
        
        # Run + readout    
        mo=build_network(modinput)
        nest.Simulate(time_ms) 
        d_spikerate[s0, 0] = (nest.GetStatus(mo['sd_ca3'],'n_events')[0]/mo['n_ca3e']/time_ms)*1000
        d_spikerate[s0, 1] = (nest.GetStatus(mo['sd_ca3i'],'n_events')[0]/mo['n_ca3i']/time_ms)*1000
        # d_spikerate2[s0,s1] = (nest.GetStatus(mo['sd_ca3i'],'n_events')[0]/mo['n_ca3i']/time_ms)*1000
        # d_spikerate[r,0]=(nest.GetStatus(mo['sd_ca3'],'n_events')[0]/mo['n_ca3e']/time_ms)*1000
        # d_spikerate[r,1]=(nest.GetStatus(mo['sd_ca3i'],'n_events')[0]/mo['n_ca3i']/time_ms)*1000
                # d_spikerate[r,2]=(nest.GetStatus(mo['sd_ca1'],'n_events')[0]/mo['n_ca1e']/time_ms)*1000

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
    

    # Return nodes etc as a dictionary
    mo.update({'time_ms':time_ms})
    return mo
    
def plot_spikesvolts(mo):
    fig = plt.figure()
    fig_ncols=4
    fig_nrows=4
    fig_plotspikes_nneurons=100
    k=1

    # Poisson Input
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('Poisson input')
    spiketimes=nest.GetStatus(mo['sd_input'], 'events')[0]['times'] # Spike times 
    neuronidlist=np.ones(len(spiketimes))  
    plt.scatter(spiketimes, neuronidlist)
    k=k+1

    # Theta node
    k=k+3

    # CA3
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('[CA3] voltage')
    voltrace=nest.GetStatus(mo['vlt_ca3'],'events')[0]['V_m']  # Vm
    plt.plot(range(len(voltrace)), voltrace)
    plt.ylim(ymax=-50, ymin=-75)
    k=k+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('[CA3] spikes')
    spiketimes=nest.GetStatus(mo['sd_ca3'], 'events')[0]['times'] # Spike times 
    neuronidlist=nest.GetStatus(mo['sd_ca3'], 'events')[0]['senders'] 
    plt.scatter(spiketimes, neuronidlist)
    try:
        plt.ylim(ymin= np.unique(neuronidlist)[0],  ymax=np.unique(neuronidlist)[0]+fig_plotspikes_nneurons)
    except:
        pass
    k=k+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('[CA3i] voltage')
    voltrace=nest.GetStatus(mo['vlt_ca3i'],'events')[0]['V_m']  # Vm
    plt.plot(range(len(voltrace)), voltrace)
    plt.ylim(ymax=-50, ymin=-75)
    k=k+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('[CA3i] spikes')
    spiketimes=nest.GetStatus(mo['sd_ca3i'], 'events')[0]['times'] # Spike times 
    neuronidlist=nest.GetStatus(mo['sd_ca3i'], 'events')[0]['senders'] 
    plt.scatter(spiketimes, neuronidlist)
    try:
        plt.ylim(ymin= np.unique(neuronidlist)[0],  ymax=np.unique(neuronidlist)[0]+fig_plotspikes_nneurons)
    except:
        pass
    k=k+1


    # CA1  
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('[CA1] voltage')
    voltrace=nest.GetStatus(mo['vlt_ca1'],'events')[0]['V_m']  # Vm
    plt.plot(range(len(voltrace)), voltrace)
    plt.ylim(ymax=-50, ymin=-75)
    k=k+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('[CA1] spikes')
    spiketimes=nest.GetStatus(mo['sd_ca1'], 'events')[0]['times'] # Spike times 
    neuronidlist=nest.GetStatus(mo['sd_ca1'], 'events')[0]['senders'] 
    plt.scatter(spiketimes, neuronidlist)
    try:
        plt.ylim(ymin= np.unique(neuronidlist)[0],  ymax=np.unique(neuronidlist)[0]+fig_plotspikes_nneurons)
    except:
        pass
    k=k+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('[CA1i] voltage')
    voltrace=nest.GetStatus(mo['vlt_ca1i'],'events')[0]['V_m']  # Vm
    plt.plot(range(len(voltrace)), voltrace)
    plt.ylim(ymax=-50, ymin=-75)
    k=k+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('[CA1i] spikes')
    spiketimes=nest.GetStatus(mo['sd_ca1i'], 'events')[0]['times'] # Spike times 
    neuronidlist=nest.GetStatus(mo['sd_ca1i'], 'events')[0]['senders'] 
    plt.scatter(spiketimes, neuronidlist)
    try:
        plt.ylim(ymin= np.unique(neuronidlist)[0],  ymax=np.unique(neuronidlist)[0]+fig_plotspikes_nneurons)
    except:
        pass
    k=k+1

    # AMY
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('[AMY] voltage')
    voltrace=nest.GetStatus(mo['vlt_amy'],'events')[0]['V_m']  # Vm
    plt.plot(range(len(voltrace)), voltrace)
    plt.ylim(ymax=-50, ymin=-75)
    k=k+1
    plt.subplot(fig_nrows, fig_ncols, k)  
    plt.title('[AMY] spikes')
    spiketimes=nest.GetStatus(mo['sd_amy'], 'events')[0]['times'] # Spike times 
    neuronidlist=nest.GetStatus(mo['sd_amy'], 'events')[0]['senders'] 
    plt.scatter(spiketimes, neuronidlist)
    try:
        plt.ylim(ymin= np.unique(neuronidlist)[0],  ymax=np.unique(neuronidlist)[0]+fig_plotspikes_nneurons)
    except:
        pass
    k=k+1


    # Display !! 
    plt.show()




##########################################################################################
##########################################################################################
######################## WHAT DO YOU WANT TO DO RIGHT NOW ? ##############################
##########################################################################################
##########################################################################################


""" [1] Parameter search  """
# so=scan_weights()  # Parameter search
# print so['d_spikerate'], so['scanpar']
# mo=so['mo']
# plot_spikesvolts(mo)

""" [2] Just run the model"""
mo=JustRun()
plot_spikesvolts(mo)




# # # Nest-inspect a particular node
# nest.raster_plot.from_device(mo['sd_ca3'])
# nest.raster_plot.show()
# nest.voltage_trace.from_device(mo['vlt_ca3i'])







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