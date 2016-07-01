"""
Example 3: A basic balanced network of excitatory and inhibitory neuronal populations.
Based upon the networks in [1] and [2]

Modified from the balanced network version already included in nest, and also 
available from http://www.nest-initiative.org/The_balanced_random_network_model.

[1] van Vreeswijk and Sompolinsky (1996)
[2] Brunel (2000)

@sjarvis, ACCN 2014
"""

import nest
import nest.raster_plot

# we need numpy to perform a little calculation
import numpy 

# and we'll use the time module to take how long the network takes to run
import time

# -----------------------------------------------
# Network constants
# -----------------------------------------------
# Network size params
order = 100
NE = 4*order  # number of excitatory neurons
NI = 1*order  # number of inhibitory neurons
Ntot = NE+NI  # total number of neurons
# as we will have a smaller inhibitory than excitatory population, we have 
# compensate inhibitory synaptic strength by a factor, referred to a g in [2]
g = 5   
# We can also gate the amount of noise that drives the network by variable eta
eta = 2

# Connectivity params
epsilon = 0.1 # connectivity of the network: 10% connectivity
CE = int(epsilon*NE)
CI = int(epsilon*NI)
Ctot = CE+CI

# Synapse parameters
tauSyn = 0.5
CMem = 250.0
tauMem = 20.0
theta  = 20.0

# The connection strength
J = 0.1 # PSP height in mV
Jcorr = 0.00483 # correction we need to make, given our choice of synapse values
J_ex = J/Jcorr
J_in = g*J
delay = 1.5

# Number of neurons we'll record from
Nrec = 100


# Input driving rate:
# threshold rate, equivalent rate of events needed to
# have mean input current equal to threshold
nu_th  = (theta * CMem) / (J_ex*CE*numpy.exp(1)*tauMem*tauSyn)
nu_ex  = eta*nu_th
p_rate = 1000.0*nu_ex*CE


runtime = 500. # the time we simulate for
dt = 0.1        # the timestep resolution


# -----------------------------------------------
# 0. Set up our environment 
# -----------------------------------------------
# Reset the kernel so that our network will be fresh
nest.ResetKernel()
# Set the resolution and overwrite any .gdf files we generate
nest.SetKernelStatus({"resolution": dt, "print_time": True,'overwrite_files':True})

# -----------------------------------------------
# 1. Define our neuron params
# -----------------------------------------------
neuron_params= {"C_m"       : CMem,
                "tau_m"     : tauMem,
                "tau_syn_ex": tauSyn,
                "tau_syn_in": tauSyn,
                "t_ref"     : 2.0,
                "E_L"       : 0.0,
                "V_reset"   : 0.0,
                "V_m"       : 0.0,
                "V_th"      : theta}
nest.CopyModel("iaf_psc_alpha", "my_neurons", neuron_params)

# -----------------------------------------------
# 2. Create our neuron populations
# -----------------------------------------------

pop_ex = nest.Create("my_neurons",NE)
pop_in = nest.Create("my_neurons",NI)

# -----------------------------------------------
# 3. Connect our populations together
# -----------------------------------------------
conn_options = {'allow_autapses': False,
                'allow_multapses':False}

nest.RandomConvergentConnect(pop_ex, pop_ex, CE, J_ex, delay, options=conn_options)
nest.RandomConvergentConnect(pop_ex, pop_in, CE, J_in, delay, options=conn_options)
nest.RandomConvergentConnect(pop_in, pop_ex, CI, J_ex, delay, options=conn_options)
nest.RandomConvergentConnect(pop_in, pop_in, CI, J_in, delay, options=conn_options)


# -----------------------------------------------
# 4. Set up our input
# -----------------------------------------------

# Create a default excitatory connection for when we start hooking up how we 
# drive and record from the network
nest.CopyModel("static_synapse","excitatory",{"weight":J_ex, "delay":delay})
 
# Create our poisson input
nest.SetDefaults("poisson_generator",{"rate": p_rate})
noise=nest.Create("poisson_generator")
# Connect it to our populations
nest.DivergentConnect(noise,pop_ex,model="excitatory")
nest.DivergentConnect(noise,pop_in,model="excitatory")



# -----------------------------------------------
# 5. Record
# -----------------------------------------------

# Create our spike detectors
espikes=nest.Create("spike_detector")
ispikes=nest.Create("spike_detector")
# Set espikes so that they print to file with the time
nest.SetStatus(espikes,[{"to_file": True, "withtime": True}])

# Connect them up to our model. Note that we're just going to record
# from the first Nrec neurons
nest.ConvergentConnect(pop_ex[:Nrec],espikes,model="excitatory")
nest.ConvergentConnect(pop_in,ispikes,model="excitatory")


# -----------------------------------------------
# 6. Run
# -----------------------------------------------
print("About to run the network ...")

startrun = time.time()
nest.Simulate(runtime)
endrun = time.time()

print("Finished running the network.")
print("Run time: %.2f s"%(endrun-startrun))
# Get information about where our file was saved to
filename = nest.GetStatus(espikes,'filenames')[0][0]
print("Saved output in %s"%filename)

# -----------------------------------------------
# 7. Show and run stats
# -----------------------------------------------
# Just an example of how we can get information directly from the spike detectors:
events_ex = nest.GetStatus(espikes,"n_events")[0]
rate_ex   = events_ex/runtime*1000.0/NE
events_in = nest.GetStatus(ispikes,"n_events")[0]
rate_in   = events_in/runtime*1000.0/NI
print "Rates for excitatory population: %.2f Hz"%rate_ex
print "Rates for inhibitory population: %.2f Hz"%rate_in

# Plot the spikes
nest.raster_plot.from_device(espikes,hist=True)
nest.raster_plot.show()

