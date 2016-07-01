import sys
sys.path.append("/Users/EleanorL/NEST/nest24/ins/lib/python2.7/site-packages")
import nest
import nest.raster_plot
import nest.voltage_trace
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

# dt=0.1
# nest.ResetKernel()



# # # Dynamics
# theta_amp=7.
# theta_freq= 70.5  # theta freq: 6-10 Hz
# theta_offset=0.
# theta_phase=0.
# weight_thetainput=0.8
# theta_dc=0.


# active_spiketimes=np.arange(1.,100.,1.)

# # ################################################################################
# """ Create nodes """ 
# nest.CopyModel("static_synapse","readout", {"weight":1., "delay": dt})  


# # Theta (spikes)
# input_theta=nest.Create("sinusoidal_poisson_generator", 1, {'dc': theta_dc, 'ac': theta_amp,  'freq': theta_freq} )
# sd_theta=nest.Create("spike_detector") # , params={"to_file": True, "withtime": True, "label":'sd_ca3'})  
# nest.Connect(input_theta, sd_theta, model='readout')

# # # Active spikes
# # input_active=nest.Create('spike_generator',1, {'spike_times': active_spiketimes})
# # sd_active=nest.Create("spike_detector") # , params={"to_file": True, "withtime": True, "label":'sd_ca3'})  
# # nest.Connect(input_active, sd_active, model='readout')

# # Run
# nest.Simulate(1000.) 






# # Nest-inspect a particular node
# mo={'sd_theta':sd_theta}
# print 'No. of spikes from theta gen: ', nest.GetStatus(mo['sd_theta'], 'n_events')[0]
# nest.raster_plot.from_device(mo['sd_theta'],hist=True)
# nest.raster_plot.show()



""" POWER ANALYSIS SLIDING WINDOW """

# # Fake raster plot
# n_neurons=900
# mo={'time_ms': 1000.}
# timebin_ms=50.
# spikeraster=npr.random_integers(0,1, [900,1000])
# #
# n_neurons=1


# """ DO A THING """
# # nodename='ca3'
# # timebin_ms=mo['time_ms']/20

# # # Get spike data in raster (row=neuron, col=timebin)
# # n_bins=int(mo['time_ms']/timebin_ms)
# # n_autocorss=n_bins+1 # Assuming 50% bin overlap
# # exec("spikedetector=mo['sd_" + nodename + "']")
# # exec "n_neurons=len(nest.GetConnections(target=mo['sd_"+nodename + "']))"
# # ss=nest.GetStatus(spikedetector, 'events')
# # spiket=np.round(ss[0]['times'])
# # spikewho=ss[0]['senders']
# # spikeraster=np.zeros([n_neurons, int(mo['time_ms'])])   # Binary spike/not for each neuron x timestep. row = neuron, col = time bin (1 ms)
# # for e in range(len(ss[0]['times'])):
# #     spikeraster[spikewho[e]-np.min(spikewho)-1, spiket[e]-1]=1

# # Autocorrelation within time bins (50% overlap)
# autocor_binstart=np.unique(np.round(np.arange(0, int(mo['time_ms']), int(timebin_ms)/2))) # Indexing the spikes stored in spikeraster
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
#         freq_thissnipbin=freq_thissnipbin+spec
#     freq_thissnipbin=freq_thissnipbin/n_neurons  # Mean freq vector of all neurons
#     powerspec[:, bs]=freq_thissnipbin # Record to overall time x freq plot. y/row=freq, x/col=timebin
# return  spikeraster, powerspec





# spiketimes=nest.GetStatus(mo['sd_theta'], 'events')[0]['times'] # Spike times 
# neuronidlist=np.ones(len(spiketimes))  
# plt.scatter(spiketimes, neuronidlist, s=fig_rasterdot_size*5)
# plt.xlim(xmax=mo['time_ms'], xmin=0)            
# plt.xticks(range(0, int(mo['time_ms'])+1, fig_xtick_ms))
# k=k+2
# plt.subplot(fig_nrows, fig_ncols, k)  # Power analysis
# plt.title('[Theta node] FreqPower')    
# plt.plot(mo['theta_power'])
# plt.xlim(xmax=mo['time_ms']/2, xmin=0)
# plt.xticks(range(0, int(mo['time_ms'])/2+1, fig_xtick_ms))
# k=k+1
# plt.subplot(fig_nrows, fig_ncols, k)  # TimexFreq power spectogram
# plt.title('[Theta node] Time x Freq spect')    
# plt.plot(mo['theta_tfpower'])
# plt.xlim(xmax=int(binsize_ms/2), xmin=0)            
# plt.xticks(range(0, int(mo['time_ms'])/2+1, fig_xtick_ms))

# thisneuronsnip=spikeraster[n, autocor_binstart[bs]:(autocor_binstart[bs]+int(timebin_ms))]
# if not thisneuronsnip.size==int(timebin_ms):
#     print 'Check snip size (sliding window power analysis)'
# thisneuronsnip_autocor=np.correlate(thisneuronsnip, thisneuronsnip, mode='full') # Autocorrelate
# thisneuronsnip_autocor=thisneuronsnip_autocor[thisneuronsnip_autocor.size/2:]
# x=thisneuronsnip_autocor  # Fourier
# spec = np.absolute(np.fft.fft(x))
# spec = spec[:len(x) / 2]  # take positive frequency components
# spec /= len(x)  # normalize (by timesteps)
# spec *= 2.0  # to get amplitudes of sine components, need to multiply by 2
# spec[0] /= 2.0  # except for the dc component





    #     mypowerspec.extend(spec)  # Add to this neuron's time-freq plot
    #     # print str(bs), 'Length=', str(len(mypowerspec))
    # autocor_spiketrain[n,:]=np.transpose(mypowerspec) # Add to group time-freq data 
# return spikeraster, autocor_spiketrain







# # Frequency analysis? 
# import numpy as np
# n_cycles = 2  # number of cycles in Morlet wavelet
# frequencies = np.arange(7, 30, 3)  # frequencies of interest
# Fs = raw.info['sfreq']  # sampling in Hz
# from mne.time_frequency import induced_power
# power, phase_lock = induced_power(epochs_data, Fs=Fs, frequencies=frequencies, n_cycles=2, n_jobs=1)

# def spiketimes2raster(mo, nodename):
#     # Compile raster, i.e. matrix of spikes (on/off). Y axis=neuron, X axis=time
    
#     exec("spikedetector=mo['sd_" + nodename + "']")
#     exec "n_neurons=len(nest.GetConnections(target=mo['sd_"+nodename + "']))"
#     ss=nest.GetStatus(spikedetector, 'events')
#     spiket=np.round(ss[0]['times'])
#     spikewho=ss[0]['senders']
#     spikeraster=np.zeros([n_neurons, int(mo['time_ms'])])   # Binary spike/not for each neuron x timestep. row = neuron, col = time bin (1 ms)
#     for e in range(len(ss[0]['times'])):
#         print spikewho[e]-np.min(spikewho)-1, ' at ', spiket[e]-1
#         spikeraster[spikewho[e]-np.min(spikewho)-1, spiket[e]-1]=1
#     return spikeraster






# def simple_frequency_spectrum(x):
#     """Simple frequency spectrum. [http://pythonhosted.org/NeuroTools/_modules/NeuroTools/analysis.html#simple_frequency_spectrum]
#     Very simple calculation of frequency spectrum with no detrending, windowing, etc, just the first half 
#     (positive frequency components) of abs(fft(x))  

#     Parameters
#     ----------
#     x : array_like   -  The input array, in the time-domain.

#     Returns
#     -------
#     spec : array_like
#         The frequency spectrum of `x`.

#     """
#     spec = np.absolute(np.fft.fft(x))
#     spec = spec[:len(x) / 2]  # take positive frequency components
#     spec /= len(x)  # normalize (by timesteps)
#     spec *= 2.0  # to get amplitudes of sine components, need to multiply by 2
#     spec[0] /= 2.0  # except for the dc component
#     return spec



# def autocorr(x):
#     result = numpy.correlate(x, x, mode='full')
#     return result[result.size/2:]

###############################################################################################
############ CODE DUMP DONT DELETE #######################################################################
###############################################################################################


# Spike generator: how to create a thing
# nest.Create('spike_generator',1, {'spike_times': [1.,2.,3.]})  # THIS WORKS

##########################################################################################
######################################  CODE DUMP ########################################
##########################################################################################

# Interrogate connections
# a= nest.GetConnections(source=mo['pope_ca3'],target=mo['popi_ca3'])

# # Nest-inspect a particular node
# nest.raster_plot.from_device(mo['sd_ca1'],hist=True)
# nest.voltage_trace.from_device(mo['vlt_ca1'])
# nest.raster_plot.show()

# Multiplying all elements in a list
# a1 = [0,1,2] 
# a1[:] = [x*3 for x in a1]   # or replace a1[:] with a2


# # Matlab-style indexing
# ss=nest.GetStatus(mo['sd_ca3'], 'events')
# spiked=np.dstack((ss[0]['senders'], ss[0]['times']))
# spiked.sort(axis=1) 
# for n in neuronidlist:    
#     whenispike=spiked[spiked[:,0] == n][:,1]    # In matrix spiked, for all rows where col 1=n, fetch data in cols 2 



""" sliding window freq analysis """
# # code here generally works, but autocor-FT'd snips must be transposed before lining up
# nodename='ca3'
# timebin_ms=mo['time_ms']/20

# # Get spike data in raster (row=neuron, col=timebin)
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
# autocor_spiketrain=np.zeros([n_neurons, int((len(autocor_binstart)-1)*timebin_ms/2)]) # Autocorrelation plots (row=neuron)
# for n in range(n_neurons):  # For all neurons
#     myspikes=spikeraster[n,:]  
#     mypowerspec=[]
#     for bs in range(len(autocor_binstart)-1):  # For all snips
#         thissnip=spikeraster[n, autocor_binstart[bs]:(autocor_binstart[bs]+int(timebin_ms))]
#         if not thissnip.size==int(timebin_ms):
#             print 'Check snip size (sliding window power analysis)'
#         thisautocor=np.correlate(thissnip, thissnip, mode='full') # Autocorrelate
#         thisautocor=thisautocor[thisautocor.size/2:]
#         x=thisautocor  # Fourier
#         spec = np.absolute(np.fft.fft(x))
#         spec = spec[:len(x) / 2]  # take positive frequency components
#         spec /= len(x)  # normalize (by timesteps)
#         spec *= 2.0  # to get amplitudes of sine components, need to multiply by 2
#         spec[0] /= 2.0  # except for the dc component
#         mypowerspec.extend(spec)  # Add to this neuron's time-freq plot
#         # print str(bs), 'Length=', str(len(mypowerspec))
#     autocor_spiketrain[n,:]=np.transpose(mypowerspec) # Add to group time-freq data 
# # return spikeraster, autocor_spiketrain



""" Native bar charst """

N = 5
menMeans = (20, 35, 30, 35, 27)
menStd =   (2, 3, 4, 1, 2)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, menMeans, width, color='r', yerr=menStd)

womenMeans = (25, 32, 34, 20, 25)
womenStd =   (3, 5, 2, 3, 3)
rects2 = ax.bar(ind+width, womenMeans, width, color='y', yerr=womenStd)

# add some text for labels, title and axes ticks
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('G1', 'G2', 'G3', 'G4', 'G5') )

ax.legend( (rects1[0], rects2[0]), ('Men', 'Women') )

# def autolabel(rects):
#     # attach some text labels
#     for rect in rects:
#         height = rect.get_height()
#         ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
#                 ha='center', va='bottom')

# autolabel(rects1)
# autolabel(rects2)

plt.show()



""" Export to Excel """




# a = [[1,2,3],[4,5,6],[7,8,9]]
# ar = np.array(a)

# import csv

# fl = open('filename.csv', 'w')

# writer = csv.writer(fl)
# writer.writerow(['label1', 'label2', 'label3']) #if needed
# for values in ar:
#     writer.writerow(values)

# fl.close()


