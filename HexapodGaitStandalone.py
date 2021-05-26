# This is a demonstration of a Central Pattern Generator coordinating hexapod walking gait
# Neurons num. 0, 2, 4 are moving left legs
# Neurons num. 1, 3, 5 are moving right legs

from brian2 import *


# Misc. parameters
walk_str = [0, 0, 0, 1, 1, 1]
walk_left = [0, 0, 0, 0, 1, 1]
walk_right = [0, 0, 1, 1, 1, 1]
run_time = 15*second
# Neuron parameters
taum = 10*ms
Ee = 0*mV
vt = -54*mV
vr = -60*mV
El = -74*mV
taue = 5*ms
eqs = '''dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
                        dge/dt = -ge / taue : 1'''
# Connectivity Map
con_map = {
    'cpg_cpg_i': [0, 1, 2, 3, 4, 5, 0, 1, 2, 3],
    'cpg_cpg_j': [0, 1, 2, 3, 4, 5, 3, 2, 5, 4],
    'cpg_inh_i': [0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5],
    'cpg_inh_j': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'cpg_eff_i': [0, 1, 2, 3, 4, 5],
    'cpg_eff_j': [0, 1, 2, 3, 4, 5],
    'inh_cpg_i': [0, 1, 4, 5, 8, 9, 12, 13, 14, 15],
    'inh_cpg_j': [1, 0, 3, 2, 5, 4, 0, 1, 2, 3],
    'inh_eff_i': [2, 3, 6, 7, 10, 11],
    'inh_eff_j': [1, 0, 3, 2, 5, 4],
    'eff_inh_i': [2, 3, 4, 5],
    'eff_inh_j': [12, 13, 14, 15]
}

# Creating CPG
group_cpg = NeuronGroup(6, eqs, threshold='v > vt',
                                           reset='v = vr', refractory=5 * ms, method='euler')
group_inh = NeuronGroup(16, eqs, threshold='v > vt',
                                           reset='v = vr', refractory=5 * ms, method='euler')
group_eff = NeuronGroup(6, eqs, threshold='v > vt',
                                           reset='v = vr', refractory=5 * ms, method='euler')
group_cpg.v = vr
group_inh.v = vr
group_eff.v = vr

syn_cpg_cpg = Synapses(group_cpg, group_cpg, model='w : volt', on_pre='v += w')
syn_cpg_inh = Synapses(group_cpg, group_inh, model='w : volt', on_pre='v += w')
syn_cpg_eff = Synapses(group_cpg, group_eff, model='w : volt', on_pre='v += w')
syn_inh_cpg = Synapses(group_inh, group_cpg, model='w : volt', on_pre='v -= w')
syn_inh_eff = Synapses(group_inh, group_eff, model='w : volt', on_pre='v -= w')
syn_eff_inh = Synapses(group_eff, group_inh, model='w : volt', on_pre='v += w')

syn_cpg_cpg.connect(i=con_map['cpg_cpg_i'], j=con_map['cpg_cpg_j'])
syn_cpg_inh.connect(i=con_map['cpg_inh_i'], j=con_map['cpg_inh_j'])
syn_cpg_eff.connect(i=con_map['cpg_eff_i'], j=con_map['cpg_eff_j'])
syn_inh_cpg.connect(i=con_map['inh_cpg_i'], j=con_map['inh_cpg_j'])
syn_inh_eff.connect(i=con_map['inh_eff_i'], j=con_map['inh_eff_j'])
syn_eff_inh.connect(i=con_map['eff_inh_i'], j=con_map['eff_inh_j'])

syn_cpg_cpg.w = 1 * volt
syn_cpg_inh.w = 1 * volt
syn_cpg_eff.w = 1 * volt
syn_inh_cpg.w = 1 * volt
syn_inh_eff.w = 1 * volt
syn_eff_inh.w = 1 * volt

mon = SpikeMonitor(group_eff)
# Creating Poisson Input
inp = PoissonGroup(6, rates=15*Hz)
syn_inp_cpg = Synapses(inp, group_cpg, on_pre='v+=0.1 * volt')
syn_inp_cpg.connect(i=[0, 1, 2, 3, 4, 5], j=walk_str)

run(run_time/3, report='text')
fig, (pl1, pl2, pl3) = subplots(3, 1)
pl1.set_title('Gait walking straight')
pl1.plot(mon.t/ms, mon.i, 'ok')
pl1.set_xlabel('Time, ms')
pl1.set_ylabel('Neuron index')

syn_inp_cpg.connect(i=[0, 1, 2, 3, 4, 5], j=walk_left)
run(run_time/3, report='text')
pl2.set_title('Gait walking to the right')
pl2.plot(mon.t/ms, mon.i, 'ok')
pl2.set_xlabel('Time, ms')
pl2.set_ylabel('Neuron index')

syn_inp_cpg.connect(i=[0, 1, 2, 3, 4, 5], j=walk_right)
run(run_time/3, report='text')
pl3.set_title('Gait walking to the left')
pl3.plot(mon.t/ms, mon.i, 'ok')
pl3.set_xlabel('Time, ms')
pl3.set_ylabel('Neuron index')
tight_layout()
show()
