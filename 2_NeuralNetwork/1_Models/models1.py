import nest

# Universal settings
dt=nest.GetKernelStatus('resolution')
nest.CopyModel("static_synapse","readout",{"weight":1., "delay": dt})  # Connect up readout


#%% Single node population

def m1_singlenodepop(modset):
    print 'Running m1_singlenodepop'   
    
    
    # Node creation (hard coded)
#    print('Hard-coded creation of neurons')
    
    pop_ca3 = nest.Create("iaf_psc_exp",1)
    sd_ca3 = nest.Create("spike_detector", params={"to_file": True, "withtime": True, "label": "sd_ca3"})
    vlt_ca3 = nest.Create("voltmeter", params={"to_file": True, "withtime": True, "label": "vlt_ca3"})
    nest.Connect(vlt_ca3, pop_ca3, model="readout")
    nest.Connect(pop_ca3, sd_ca3, model="readout")
#    
#    pop_ca1 = nest.Create("iaf_psc_exp",1)
#    pop_amy = nest.Create("iaf_psc_exp",1)
#    
#    n
#    
##    
#    # Set up nodes + readouts (# NodeSettings in modset[0])
#    nodenames=list(modset[0].viewkeys())
#    for n in range(len(nodenames)):
#        exec("print 'Creating node '+ nodenames[n]")
##        exec('pop_' + nodenames[n] + ' = nest.Create("' + modset[0][nodenames[n]][1] + '",' + str(modset[0][nodenames[n]][2] ) + ')')
#        print('sd_' + nodenames[n] + ' = nest.Create("spike_detector", params={"to_file": True, "withtime": True, "label": "sd_' + nodenames[n][0:]+'"})'  )
#        print('vlt_' + nodenames[n] + ' = nest.Create("voltmeter", params={"to_file": True, "withtime": True, "label": "vlt_' + nodenames[n][0:]+'"})'  )
#        print('nest.Connect(vlt_' + nodenames[n][0:] + ', pop_' + nodenames[n][0:] + ', model="readout")')  # Connect readouts
#        print('nest.Connect(pop_' + nodenames[n][0:] + ', sd_' + nodenames[n][0:] + ', model="readout")')
#    for n in range(len(nodenames)):
#        exec("nodespex= modset[0]['" + nodenames[n][0:] + "'][3]")   # Alter node properties as requested    
#        spex2change=list(nodespex.viewkeys())
#        for s in range(len(spex2change)): 
#            exec("nest.SetStatus(pop_" + nodenames[n][0:]  +  ", {'" + spex2change[s] + "':" + " nodespex['"+spex2change[s]+ "']}) ")
            
#    nest.GetStatus(pop_ca3)
#    nest.GetStatus(pop_ca1)
#    nest.GetStatus(pop_amy)
        
#    # Single-node input/modulatory nodes + Connected to targets 
#    nodenames=list(modset[1].viewkeys())
#    for n in range(len(nodenames)):        
#        print 'Creating input node '+ nodenames[n]
#        exec('input_' + nodenames[n][0:] + ' = nest.Create("' + nodenames[n] + '",1)')
#        nodespex=modset[1][nodenames[n]][0]
#        spex2change=list(nodespex.viewkeys())
#        for s in range(len(spex2change)): 
#            exec("nest.SetStatus(input_" + nodenames[n]  +  ", {'" + spex2change[s] + "':" + " nodespex['"+ spex2change[s]+ "']}) ")        
#        nodes2connect=list(modset[1][nodenames[n]][1].viewkeys())  # Connect up
#        for n2c in range(len(nodes2connect)): 
#            connecspex=modset[1][nodenames[n]][1][nodes2connect[n2c]]
#            print("nest." + connecspex[0] + "(input_" + nodenames[n] + "," + "pop_" + nodes2connect[n2c] + "," + str(connecspex[1]) + "," + str(connecspex[2]) + ")")
#            
#            
##            error seems to be here in line 37!!! 
##            fix this, then auto-set up node connections


    # Connect up nodes
    print 'done setting up model'




    
#    units={'ca1':[pop_ca1, sd_ca1, vlt_ca1], 'ca3':[pop_ca3, sd_ca3, vlt_ca3], 'amy':[pop_ca1, sd_ca1, vlt_ca1]}
#    return units




#%%   
        
        
#        
#        
#        
#        # Create node
#        pop_ca1=Create(type_ca1, n_ca1)
#        pop_ca3=Create(type_ca3, n_ca3)
#        pop_amy=Create(type_amy, n_amy)
#        input_ca3=nest.Create("poisson_generator", 1, {'rate': 140000., 'start': 0., } )  # single node input
#        theta_gen=nest.Create("ac_generator", 1, {'amplitude': theta_amp, 'frequency': theta_freq} ) 
#        
        
#        
#        # Connections (Source, Target, Weight, Delay)
#        nest.CopyModel("static_synapse","ex_all",{"weight":1., "delay":1.})
#        nest.CopyModel("static_synapse","ex_input",{"weight":3., "delay":1.})
#        nest.CopyModel("static_synapse","ex_thetamod", {"weight":2., "delay":1.})
#        nest.DivergentConnect(input_ca3, pop_ca3,  model='ex_input')
#        nest.Connect(pop_ca3, pop_ca1, model='ex_all')
#        nest.Connect(pop_amy, pop_ca1, model='ex_all')
#        nest.Connect(pop_ca3, pop_amy, model='ex_all')
#        #nest.Connect=(theta_gen, pop_ca3, model='ex_thetamod')
#        
#        # Run + Output from model fxn
#        nest.Simulate(time_ms)    # RUN 
#   authorsConnect?


#%% Single node population
#
#def m1_singlenodepop(modset):
#    print 'Running m1_singlenodepop'    
#    
#    # Define nodes (# NodeSettings in 0)
##    nodenames=list(modset[0].viewkeys())
#    p_test=nest.Create('iaf_psc_alpha',1, params={'V_m':1., 'V_th':10.})
#    p_test2=nest.Create('iaf_psc_alpha',1, params={'V_m':2., 'V_th':20.})
##    return {'first': p_test, 'second': p_test2}
#    return p_test, p_test2
##    
