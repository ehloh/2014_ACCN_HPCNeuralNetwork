""" Old stype time freq analysis (sliding window) """


def spikes2powerspec_timebins(mo, nodename, timebin_ms):

    # Get spike data in raster (row=neuron, col=timebin)
    n_bins=int(mo['time_ms']/timebin_ms)
    n_autocorss=n_bins+1 # Assuming 50% bin overlap
    exec("spikedetector=mo['sd_" + nodename + "']")
    exec "n_neurons=len(nest.GetConnections(target=mo['sd_"+nodename + "']))"
    ss=nest.GetStatus(spikedetector, 'events')
    spiket=np.round(ss[0]['times'])
    spikewho=ss[0]['senders']
    spikeraster=np.zeros([n_neurons, int(mo['time_ms'])])   # Binary spike/not for each neuron x timestep. row = neuron, col = time bin (1 ms)
    for e in range(len(ss[0]['times'])):
        spikeraster[spikewho[e]-np.min(spikewho)-1, spiket[e]-1]=1

    # Autocorrelation within time bins (50% overlap)
    # #   Details on power spec function: http://matplotlib.org/api/mlab_api.html#matplotlib.mlab.psd
    bin_starts=np.unique(np.round(np.arange(0, int(mo['time_ms']), int(timebin_ms)/(1/overlap)))) # Indexing the spikes stored in spikeraster
    n_bins = np.sum(bin_starts<1000.-timebin_ms)  # Only complete bins are valid
    powerspec=np.zeros((n_freq, n_bins))   # x/col=time, y/row=freq
    for bs in range(n_bins):  # For all (timebinned) snip
        freqpower_thissnip=np.zeros((n_freq, n_neurons))  # y/row=freq, x/col=neuron
        for n in range(n_neurons):  # For all neurons
            tn_snip=spikeraster[n, bin_starts[bs]:(bin_starts[bs]+int(timebin_ms))]
            if not tn_snip.size==int(timebin_ms):
                print 'Check snip size (sliding window power analysis)'
            tn_snip_autocor=np.correlate(tn_snip, tn_snip, mode='full') # Autocorrelate
            tn_snip_autocor=tn_snip_autocor[tn_snip_autocor.size/2:]
            x=tn_snip_autocor
            # x=tn_snip
            power, freq=plt.psd(x, NFFT=int(timebin_ms), Fs=1, Fc=0, detrend=plt.mlab.detrend_none, window=plt.mlab.window_hanning, noverlap=int(overlap*int(timebin_ms)), pad_to=None, sides='default', scale_by_freq=None)
            # WHY is the window what it is? Why am I telling the function what the overlap is - that seems to be counted outside?
            freqpower_thissnip[:, n]=power.transpose()
        powerspec[:, bs]=freqpower_thissnip.mean(axis=1) 
        plt.close()
    # powerspec=powerspec.transpose()
    plt.imshow(powerspec)
    plt.show()

    return  spikeraster, powerspec

""" Better plots """
# def plot_spikesvolts_new(mo):
#     fig = plt.figure(figsize=(18,8))
#     fig_ncols=11
#     fig_nrows=4
#     fig_plotspikes_nneurons=100
#     fig_rasterdot_size=2
#     fig_histbinsize=20
#     fig_xtick_ms=500
#     k=1
#     try:
#         fig_xtick_tfbins=int(np.shape(mo['theta_tfpower'])[1])
#     except:
#         pass

#     # Poisson Input
#     plt.subplot(fig_nrows, fig_ncols, k)  
#     plt.title('Poisson noise')
#     spiketimes=nest.GetStatus(mo['sd_noiseinput'], 'events')[0]['times'] # Spike times 
#     neuronidlist=np.ones(len(spiketimes))  
#     plt.scatter(spiketimes, neuronidlist, s=fig_rasterdot_size)
#     plt.xlim(xmax=mo['time_ms'], xmin=0)
#     plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
#     k=k+1

#     # Active input
#     plt.subplot(fig_nrows, fig_ncols, k)      
#     plt.title('Active input spikes')
#     spiketimes=nest.GetStatus(mo['sd_active'], 'events')[0]['times'] # Spike times 
#     neuronidlist=np.ones(len(spiketimes))  
#     plt.scatter(spiketimes, neuronidlist, s=fig_rasterdot_size*2)
#     plt.xlim(xmax=mo['time_ms'], xmin=0)            
#     plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
#     k=k+1

#     # Theta node
#     plt.subplot(fig_nrows, fig_ncols, k)  
#     plt.title('Theta spikes')
#     try:
#         # voltrace=nest.GetStatus(mo['vlt_theta'],'events')[0]['V_m']  # Vm
#         # plt.plot(range(len(voltrace)), voltrace)
#         spiketimes=nest.GetStatus(mo['sd_theta'], 'events')[0]['times'] # Spike times 
#         neuronidlist=np.ones(len(spiketimes))  
#         plt.scatter(spiketimes, neuronidlist, s=fig_rasterdot_size*5)
#         plt.xlim(xmax=mo['time_ms'], xmin=0)            
#         plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
#         k=k+1
#         plt.subplot(fig_nrows, fig_ncols, k)  # Power analysis
#         plt.title('[ThetaIn] FreqPower')    
#         plt.plot(mo['theta_power'])
#         plt.xlim(xmax=mo['time_ms']/2, xmin=0)
#         plt.xticks(range(0, int(mo['time_ms'])/2+1, fig_xtick_ms))
#         k=k+1
#         plt.subplot(fig_nrows, fig_ncols, k)  # TimexFreq power spectogram
#         plt.title('[ThetaIn] TimeFreq Spect')    
#         plt.imshow(mo['theta_tfpower'])
#         plt.xlim(xmax=fig_xtick_tfbins, xmin=0)
#     except:
#         pass

#     fig_popnodes2plot=['ca3', 'ca1', 'amy']
#     for n in range(len(fig_popnodes2plot)):
#         k=fig_ncols*(n+1)+1

#         # Fetch data (excitatory)
#         exec("voltrace=nest.GetStatus(mo['vlt_" + fig_popnodes2plot[n] + "'],'events')[0]['V_m']")  # Vm
#         exec("spiketimes=nest.GetStatus(mo['sd_" + fig_popnodes2plot[n] + "'], 'events')[0]['times']")  # Spike times
#         neuronidlist=np.ones(len(spiketimes))  
#         try:
#             exec("freqpower_alltime=mo['" + fig_popnodes2plot[n] + "_power']")
#         except:
#             freqpower_alltime=[0]
#         try:
#             exec("timefreq_powerplot=mo['" + fig_popnodes2plot[n] + "_tfpower']")
#         except:
#             timefreq_powerplot=[0]

#         # Vm
#         plt.subplot(fig_nrows, fig_ncols, k)  
#         plt.title('['+ fig_popnodes2plot[n] +'] Vm')
#         plt.plot(range(len(voltrace)), voltrace)
#         plt.ylim(ymax=-50, ymin=-75)
#         plt.xlim(xmax=mo['time_ms'], xmin=0)
#         plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
        
#         # Spikes
#         plt.subplot(fig_nrows, fig_ncols, k+1)  
#         plt.title('Spikes')
#         plt.scatter(spiketimes, neuronidlist, s=fig_rasterdot_size)
#         if not spiketimes:
#             plt.xlim(xmax=mo['time_ms'], xmin=0)
#             plt.ylim(ymin= np.unique(neuronidlist)[0],  ymax=np.unique(neuronidlist)[0]+fig_plotspikes_nneurons)
#             plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))

#         # Spike histogram
#         plt.subplot(fig_nrows, fig_ncols, k+2)  
#         plt.title('Spike hist')    
#         plt.hist(spiketimes, bins = range(int(np.min(spiketimes)),  int(np.max(spiketimes)), fig_histbinsize))
#         if not spiketimes:
#             plt.xlim(xmax=mo['time_ms'], xmin=0)
#             plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
        
#         # Freq power whole epoch
#         try:
#             plt.subplot(fig_nrows, fig_ncols, k+3)
#             plt.title('FreqPower')    
#             plt.plot(freqpower_alltime)
#             plt.xlim(xmax=mo['time_ms']/2, xmin=0)
#             plt.xticks(range(0, int(mo['time_ms'])/2+1, fig_xtick_ms))
#         except:
#             pass

#         # Time-Frequency spectrogram
#         try:
#             plt.subplot(fig_nrows, fig_ncols, k+4)
#             plt.title('Time x Freq spect')    
#             plt.imshow(timefreq_powerplot)
#             plt.xlim(xmax=int(np.shape(timefreq_powerplot)[1]), xmin=0)
#         except:
#             pass


#     # Display !! 
#     plt.subplots_adjust(left=0.05, bottom=None, right=0.99, top=None, wspace=0.7, hspace=0.5)
#     plt.show()



# def spikes2powerspec_alltime(mo, nodename):

#     # Get spike data in raster (row=neuron, col=timebin)
#     exec("spikedetector=mo['sd_" + nodename + "']")
#     exec "n_neurons=len(nest.GetConnections(target=mo['sd_"+nodename + "']))"
#     ss=nest.GetStatus(spikedetector, 'events')
#     spiket=np.round(ss[0]['times'])
#     spikewho=ss[0]['senders']
#     spikeraster=np.zeros([n_neurons, int(mo['time_ms'])])   # Binary spike/not for each neuron x timestep. row = neuron, col = time bin (1 ms)
#     for e in range(len(ss[0]['times'])):
#         spikeraster[spikewho[e]-np.min(spikewho)-1, spiket[e]-1]=1

#     # Autocorrelation
#     autocor_spiketrain=np.zeros([n_neurons, int(mo['time_ms'])]) # Autocorrelation plots (row=neuron)
#     for n in range(n_neurons):
#         ast=np.correlate(spikeraster[n,:],spikeraster[n,:], mode='full')
#         autocor_spiketrain[n,:]=ast[ast.size/2:]

#     # Fourier
#     # Simple frequency spectrum [http://pythonhosted.org/NeuroTools/_modules/NeuroTools/analysis.html#simple_frequency_spectrum]
#     # Very simple calculation of frequency spectrum with no detrending, windowing, etc, just the first half 
#     # (positive frequency components) of abs(fft(x))  
#     #
#     # Parameters
#     # ----------
#     # x : array_like   -  The input array, in the time-domain.
#     #
#     # Returns
#     # -------
#     # spec : array_like
#     #     The frequency spectrum of `x`.
#     powerspex=np.zeros([n_neurons, int(mo['time_ms'])/2]) # Power spectrum data (row=neuron)
#     for n in range(n_neurons):
#         x=autocor_spiketrain[n,:]
#         spec = np.absolute(np.fft.fft(x))
#         spec = spec[:len(x) / 2]  # take positive frequency components
#         spec /= len(x)  # normalize (by timesteps)
#         spec *= 2.0  # to get amplitudes of sine components, need to multiply by 2
#         spec[0] /= 2.0  # except for the dc component
#         powerspex[n,:]=spec
#     poppower=powerspex.mean(axis=0)
#     return spikeraster, autocor_spiketrain, powerspex, poppower


