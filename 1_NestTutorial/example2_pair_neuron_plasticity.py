"""
Example 2: Pair of neurons connected with a Tsodyks synapse

We extend the previous example to include two neurons connected together
by a facilitating / depressing synapse.

This example is based on the NEST tutorial of Jan Moren (OIST 2014)

@sjarvis, ACCN 2014
"""

import nest 
import nest.voltage_trace

nest.ResetKernel()

# -----------------------------------------------
# 1. Define our pre and post neurons
# -----------------------------------------------
neuron1 = nest.Create("iaf_neuron")
neuron2 = nest.Create("iaf_neuron")
nest.SetStatus(neuron1, {"I_e":376.0})


# -----------------------------------------------
# 2. Connect using a depressing synapse
# -----------------------------------------------
# To see all the types of synapses, use:
# nest.Models(mtype='synapses')

# We're going to use a tsodyks_synapse for the connection between our two neurons.
# We could go through and use this line 
# nest.Connect(neuron1, neuron2, model='tsodyks_synapse')
# which would give the default values for the tsodyks_synapse.

# To see what the default values are, and a little bit about the synapse
# nest.help('tsodyks_synapse')

# However, we want to define our own values and set it to be a depressing synapse. 
# In this case, it's often convenient to declare parameter dictionaries separately like
# this:

syn_param_dep = {"weight": 20.0,    # pA synaptic strength
             "tau_psc" : 2.0,   # ms synaptic rise time 
             "tau_rec" : 800.0, # ms time constant of recovery
             "tau_fac": 0.0,    # ms time constant of facilitation
             "U": 0.5}          # probability of quantal release

# We can also define parameter settings for facilitation rather than depression. 

syn_param_fac = {"weight": 20.0,    # pA synaptic strength
             "tau_psc" : 2.0,   # ms synaptic rise time 
             "tau_rec" : 0.0,   # ms time constant of recovery
             "tau_fac": 1000.0, # ms time constant of facilitation
             "U": 0.5}          # probability of quantal release

# We can now go through and define new models for our versions of the 
# depressing and facilitating synapses:
nest.CopyModel('tsodyks_synapse', 'tsodyks_dep', syn_param_dep)
nest.CopyModel('tsodyks_synapse', 'tsodyks_fac', syn_param_fac)
# Connect takes optional arguments for our synapse model.
nest.Connect(neuron1, neuron2, syn_spec='tsodyks_dep')
# If you want to use facilitation instead, try:
#nest.Connect(neuron1, neuron2, syn_spec='tsodyks_fac')


# -----------------------------------------------
# 3. Set up our recording device 
# -----------------------------------------------
mmeter = nest.Create("multimeter")
nest.SetStatus(mmeter, {"record_from": ["V_m"], "withtime": True})

nest.Connect(mmeter, neuron2)



# -----------------------------------------------
# 4. Simulate
# -----------------------------------------------
nest.Simulate(1000.0)


# -----------------------------------------------
# 5. Plot!
# -----------------------------------------------
nest.voltage_trace.from_device(mmeter)
nest.voltage_trace.show()

