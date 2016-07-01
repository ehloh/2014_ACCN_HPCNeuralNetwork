"""
Example 1: Single neuron with input and output

We start off with the simplest network model possible: a single neuron with 
a constant current that spikes. We set up and record from the membrane voltage
and the spikes emitted.

This example is based on the NEST tutorial sheets and previous NEST tutorial 
exercises from ACCN / OIST / LASCON.

@sjarvis, ACCN 2014
"""


import nest
import nest.raster_plot
import nest.voltage_trace
# Reset the kernel, so that we create a fresh network each time.
# Otherwise, running this code multiple times will start the simulation time
# where the previous simulation time ended. 
nest.ResetKernel()


# -----------------------------------------------
# 1. Create our neuron
# -----------------------------------------------
neuron = nest.Create("iaf_neuron")
# Querying our model:
print nest.GetStatus(neuron)
# Set status of threshold
#nest.SetStatus(neuron,{'V_th':-55.5})
# Check that it's set correctly
print nest.GetStatus(neuron,'V_th')

# -----------------------------------------------
# 2. Set up input
# -----------------------------------------------
# We can provide a new object to provide input (and will in the next example!)
# but for the moment let's go through and simply set the background current for 
# neuron to a value that will cause it to spike:
nest.SetStatus(neuron,{'I_e':376.0})


# -----------------------------------------------
# 3. Record from the neuron
# -----------------------------------------------
# Using a spike detector, we'll be able to record spikes:
sdetector = nest.Create("spike_detector")

# To record voltage, we'll use a multimeter
mmeter = nest.Create("multimeter")
nest.SetStatus(mmeter, {"record_from": ["V_m"], "withtime": True})


# -----------------------------------------------
# 4. Connect our recording devices to our neuron:
# -----------------------------------------------
nest.Connect(neuron,sdetector)
nest.Connect(mmeter,neuron)


# -----------------------------------------------
# 5. Simulate
# -----------------------------------------------
sim_time = 1000
nest.Simulate(sim_time)

# -----------------------------------------------
# 6. Plot our results
# -----------------------------------------------
# View our membrane voltage
nest.voltage_trace.from_device(mmeter)
# View our spikes
nest.raster_plot.from_device(sdetector)

nest.raster_plot.show()


# We can also go through and get the spike times ourselves:
# The code below does effectively the same thing as the raster_plot above. It
# is meant to illustrate how you can do this kind of visualization yourself.

#dSD = nest.GetStatus(sdetector)[0]
#evs = dSD['events']['senders']
#ts = dSD['events']['times']
