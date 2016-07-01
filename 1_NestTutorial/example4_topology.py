"""
Example 4: Topology network with input and recurrent connectivity

This example is based on the NEST tutorial of Jan Moren (OIST 2014)

@sjarvis, ACCN 2014
"""

import nest
import nest.raster_plot
# This imports the topology module that is required for 
# defining structured connectivity
import nest.topology as tp    

import numpy as np
import pylab

nest.ResetKernel()

simtime = 1000.

# Some definitions for our network:
nest.CopyModel("static_synapse","excitatory_top",{"weight":1., "delay":1.})
nest.CopyModel("poisson_generator","high_noise",{"rate":10000.})


# The layer parameter dictionary has many other parameters to set the size and
# position of the layer; 3D positions; boundary conditions and so on.
# For the moment, we'll start simple by considering a grid.
# We set 'columns' and 'rows': 

neuronlayer = tp.CreateLayer(                # Grid layer
    {   'columns'  : 11, 
        'rows'     : 11, 
        'elements' : 'iaf_neuron'})

# We set the position of our single input explicitly; that makes it a free
# layer

noiselayer = tp.CreateLayer(                # Free layer
    {   'positions': [[0.0, 0.0]],        
        'elements' : 'high_noise'})

# We first define the connectivity for the input to the network. This is done
# using a dictionary of parameters:
# topology connections can specify the possible targets through the mask; and
# the connection probability kernel, the connection weights and the delays
# with a number of distributions. In addition you can specify whether multiple
# connections to the same target and connections back to the originating node
# are allowed.

conndict = {'connection_type': 'divergent',
        'mask': {'circular': {'radius': 0.5}},
        'kernel': {'gaussian': {'p_center': 1.0, 'sigma': 0.17}}, # the probability of making a connection
        'weights': 10.0, # the weight: a constant value
        'delays':1.5} # the delay: also a constant value

# Connect input to network
tp.ConnectLayers(noiselayer, neuronlayer, conndict)

# We're going to define a second parameter set for 
recurrent_conn = {'connection_type':'divergent',
                  'mask':{'circular': {'radius': 1.5}},
                  'kernel': {'gaussian': {'p_center': 1.0, 'sigma': 0.17}},
                  # here we'll set our weights to be distance dependent:
                  'weights': {'gaussian': {'p_center': 80.0, 'sigma': 0.87}}, 
                  'delays':1.5}
                  
# Connect neurons to other neurons
tp.ConnectLayers(neuronlayer, neuronlayer, recurrent_conn)


# Set up our spike detector:
spikes=nest.Create("spike_detector")
nest.ConvergentConnect(nest.GetNodes(neuronlayer)[0],spikes,model="excitatory_top")
    

def query_activity(spikes):
    """ Let's find the time to first spike: """
    events = nest.GetStatus(spikes,"events")[0]
    spiketimes = events['times']
    neuron_ids = events['senders']
    # find first index of each sender
    latency_indices = []
    for neuron in nest.GetNodes(neuronlayer)[0]:
        latency = np.where(neuron_ids==neuron)
        if len(latency[0])==0:
	    latency_indices.append(None)
	else:
	    latency_indices.append(np.where(neuron_ids==neuron)[0][0])
    
    # and then use this to find the corresponding time
    latency_times = [spiketimes[lat] if  lat is not None else simtime for lat in latency_indices]
    # Extension: find the position of each neuron (from neuron_layer) and plot 
    # the latency to first spike of the neuron
    return latency_indices,latency_times
    

def plot_topology():
    
    noise = tp.FindNearestElement(noiselayer, [0.0, 0.0])
    
    fig = tp.PlotLayer(neuronlayer, nodecolor="orange", nodesize=5)
    
    tp.PlotTargets(noise, neuronlayer, fig=fig,
        mask=conndict['mask'], 
        kernel = conndict['kernel'],
        src_size=250, 
        tgt_color='blue', 
        tgt_size=25,
        kernel_color="lightgreen", 
        mask_color='lightblue')
    
    pylab.show()


    
nest.Simulate(simtime)
# plot the topology:
plot_topology()
# create a raster plot of the data
nest.raster_plot.from_device(spikes,hist=True)
nest.raster_plot.show()
    