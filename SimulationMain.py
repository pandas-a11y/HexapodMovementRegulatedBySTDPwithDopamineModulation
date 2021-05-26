"""
Project by Morozova E. S.
Implementation of a Central Pattern Generator coordinating hexapod walking gait
regulated by a spiking neuron network with STDP and dopamine reward modulation
Created using Python 3.8, brian2 v. 2.4.2

Using information from papers:
"Six-legged walking in insects: how CPGs, peripheral feedback, and
descending signals generate coordinated and adaptive motor rhythms"
Salil S. Bidaye, Till Bockemühl and Ansgar Büschges
J Neurophysiol 119: 459 – 475, 2018. First published October 25, 2017

"A spiking neural program for sensorimotor control during foraging in flying insects"
Hannes Rapp, Paul Nawrot
PNAS November 10, 2020 117 (45) 28412-28421. First published October 29, 2020;

"Dopamine signalling in locusts and other insects"
Heleen Verlinden
Insect Biochem Mol Biol. 2018 Jun;97:40-52. doi: 10.1016/j.ibmb.2018.04.005. Epub 2018 Apr 20.

"Solving the Distal Reward Problem through Linkage of STDP and Dopamine Signaling"
Eugene M. Izhikevich, Cerebral cortex 17, no. 10 (2007): 2443-2452.

Results: Raster plots of neuron activities in main and efferent groups,
number of total food collected per simulation cycle
"""
from brian2 import *


@implementation('numpy', discard_units=True)
@check_units(obj_angle=1, target_x=1, target_y=1, result=1)
def findAngle(obj_angle, target_x, target_y):
    return remainder((((obj_angle - (arctan(target_y / target_x))) * 180/pi)), 360);


@implementation('numpy', discard_units=True)
@check_units(x1=1, x2=1, y1=1, y2=1, result=1)
def findDistance(x1, x2, y1, y2):
    return ((x1 - x2)**2 + ((y1 - y2)**2))**0.5;


@implementation('numpy', discard_units=True)
@check_units(deg=1, result=1)
def inputResolution(deg):
    return 360 // deg


@implementation('numpy', discard_units=True)
@check_units(angle_target=1, result=1)
def getDirection(angle_target, n_neurons):
    return angle_target // (360 // n_neurons);


##################################################
# Parameters
##################################################
# Simulation and environment parameters

# Dopamine simulation parameters adapted from paper
# "Solving the Distal Reward Problem through Linkage of STDP and Dopamine Signaling"
# by Eugene M. Izhikevich, Cerebral cortex 17, no. 10 (2007): 2443-2452.

# Neuron parameters adapted from paper
# "Competitive Hebbian learning through spike-timing-dependent synaptic plasticity"
# by Sen Song, Kenneth D. Miller and L. F. Abbott, Nature Neuroscience 3, no. 9 (2000): 919-926.

# Auxiliary environment parameters
simulation_duration = 120 * second
food_location = {'food_x': [0.9, 0.1, 0.4, 0.3, -1.3, 1, 2, -3, 0.5, -1.2],
                 'food_y': [1, 1, -0.2, 4, 0.8, -1.3, -2, -1.4, 1.1, -1.3]}
food_amount = len(food_location['food_x'])
r_feed = 1    # 'eating' radius
r_see = 10    # 'seeing' radius
r_mov = 0.33    # movement distance
angle_turn_left_leg = 5    # degrees
angle_turn_right_leg = -5    # degrees
aux_n_inputs = inputResolution(30)

# Neuron equations
eqs = '''dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
                        dge/dt = -ge / taue : 1'''

# Coefficient for gaussian connectivity in syn_main
coef = int(np.ceil(np.sqrt(np.log(0.001) / -0.1)))

# Connectivity map for CPG
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

# Neuron parameters
n_num = 30
taum = 10*ms
Ee = 0*mV
vt = -54*mV
vr = -60*mV
El = -74*mV
taue = 5*ms

# STDP parameters
taupre = 20*ms
taupost = taupre
gmax = .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax

# Dopamine reward parameters
tauc = 1000*ms
taud = 200*ms
taus = 1*ms
epsilon_dopa = 5e-3

##################################################
# Set up
##################################################
# Auxiliary simulation and environment set up
print("Setting up stage...")

# Food and location processing groups
aux_food = NeuronGroup(food_amount, model='''food_x : 1
                                    food_y : 1''')
aux_food.set_states(food_location)

aux_locating = NeuronGroup(1, '''location_x : 1 
                            location_y : 1 
                            mov_angle : 1 
                            ''')
aux_locating.location_x = 0
aux_locating.location_y = 0
aux_locating.mov_angle = 0

# Environment processing group set up
aux_searching = NeuronGroup(food_amount, model='''location_x : 1 (linked)
                                                location_y : 1 (linked)
                                                food_x : 1 (linked)
                                                food_y : 1 (linked)
                                                mov_angle : 1 (linked)
                                                angle_food : 1
                                                distance_food : 1
                                                eaten : 1
                            ''',
                            events={'food_found': 'findDistance(food_x, location_x, food_y, location_y) <= r_see',
                                    'food_eaten': 'findDistance(food_x, location_x, food_y, location_y) <= r_feed'})
aux_searching.location_x = linked_var(aux_locating, name='location_x', index=repeat(0, food_amount))
aux_searching.location_y = linked_var(aux_locating, name='location_y', index=repeat(0, food_amount))
aux_searching.food_x = linked_var(aux_food, name='food_x')
aux_searching.food_y = linked_var(aux_food, name='food_y')
aux_searching.mov_angle = linked_var(aux_locating, name='mov_angle')
aux_searching.angle_food = repeat(0, food_amount)
aux_searching.distance_food = repeat(100, food_amount)
aux_searching.eaten = 0

# Event processors
aux_found_mon = EventMonitor(aux_searching, 'food_found', variables=['angle_food', 'distance_food'])
aux_searching.run_on_event('food_found', '''angle_food = findAngle(mov_angle, food_x, food_y) 
                                            distance_food = findDistance(location_x, food_x, location_y, food_y)
                                ''')
aux_eaten_mon = EventMonitor(aux_searching, 'food_eaten', variables=['eaten'])
aux_searching.run_on_event('food_eaten', '''eaten =eaten + 1
                                            food_x = food_x + (2 * rand() - 1)
                                            food_y = food_y + (2 * rand() - 1)''')

# Input (angle, distance) processing group
aux_input = NeuronGroup(aux_n_inputs, eqs, threshold='v > vt',
                                           reset='v = vr', refractory=5 * ms, method='exact')
aux_input.v = vr
aux_syn_input = Synapses(aux_searching, aux_input, model='''''', on_pre={'inp': 'v_post += (j==getDirection('
                                                                                'angle_food_pre, aux_n_inputs)) * 5 *'
                                                                                ' (r_see - (r_see // ('
                                                                                'distance_food_pre + 0.001))) * volt'},
                on_event={'inp': 'food_found'})
aux_syn_input.connect()

# Main neuron group set up
group_main_neurons = NeuronGroup(n_num, eqs,
                      threshold='v>vt', reset='v = vr',
                      method='exact')
group_main_neurons.v = vr

# Creating Gaussian connectivity, STDP
syn_main = Synapses(group_main_neurons, group_main_neurons, model='''mode: 1
                         dc/dt = -c / tauc : 1 (clock-driven)
                         dd/dt = -d / taud : 1 (clock-driven)
                         ds/dt = mode * c * d / taus : 1 (clock-driven)
                         dApre/dt = -Apre / taupre : 1 (event-driven)
                         dApost/dt = -Apost / taupost : 1 (event-driven)''',
                   on_pre='''ge += s
                          Apre += dApre
                          c = clip(c + mode * Apost, -gmax, gmax)
                          s = clip(s + (1-mode) * Apost, -gmax, gmax)
                          ''',
                   on_post='''Apost += dApost
                          c = clip(c + mode * Apre, -gmax, gmax)
                          s = clip(s + (1-mode) * Apre, -gmax, gmax)
                          ''',
                   method='euler')
syn_main.connect(j='k for k in range(i - coef, i + coef) if rand()<exp(-0.1*(i-j)**2)', skip_if_invalid=True)
syn_main.mode = 0
syn_main.s = 1e-10
syn_main.c = 1e-10
syn_main.d = 0

# Auxiliary modulator group set up
group_modulator = NeuronGroup(6, eqs, threshold='v > vt',
                                           reset='v = vr', refractory=5 * ms, method='exact')
group_modulator.v = vr

# CPG set up
group_cpg = NeuronGroup(6, eqs, threshold='v > vt',
                                           reset='v = vr', refractory=5 * ms, method='exact')
group_inh = NeuronGroup(16, eqs, threshold='v > vt',
                                           reset='v = vr', refractory=5 * ms, method='exact')
group_eff = NeuronGroup(6, eqs, threshold='v > vt',
                                           reset='v = vr', refractory=5 * ms, method='exact')
group_cpg.v = vr
group_inh.v = vr
group_eff.v = vr

syn_cpg_cpg = Synapses(group_cpg, group_cpg, model='w : volt', on_pre='v += w')
syn_cpg_inh = Synapses(group_cpg, group_inh, model='w : volt', on_pre='v += w')
syn_cpg_eff = Synapses(group_cpg, group_eff, model='w : volt', on_pre='v += w')
syn_inh_cpg = Synapses(group_inh, group_cpg, model='w : volt', on_pre='v -= w')
syn_inh_eff = Synapses(group_inh, group_eff, model='w : volt', on_pre='v -= w')
syn_eff_inh = Synapses(group_eff, group_inh, model='w : volt', on_pre='v += w')

# Connecting CPG, inhibitory, efferent group
syn_cpg_cpg.connect(i=con_map['cpg_cpg_i'], j=con_map['cpg_cpg_j'])
syn_cpg_inh.connect(i=con_map['cpg_inh_i'], j=con_map['cpg_inh_j'])
syn_cpg_eff.connect(i=con_map['cpg_eff_i'], j=con_map['cpg_eff_j'])
syn_inh_cpg.connect(i=con_map['inh_cpg_i'], j=con_map['inh_cpg_j'])
syn_inh_eff.connect(i=con_map['inh_eff_i'], j=con_map['inh_eff_j'])
syn_eff_inh.connect(i=con_map['eff_inh_i'], j=con_map['eff_inh_j'])

# Setting scale-appropriate weights
syn_cpg_cpg.w = 200 * mV
syn_cpg_inh.w = 100 * mV
syn_cpg_eff.w = 200 * mV
syn_inh_cpg.w = 100 * mV
syn_inh_eff.w = 100 * mV
syn_eff_inh.w = 100 * mV

# Connecting networks
# Connection main neuron group to auxiliary modulator group
syn_main_mod = Synapses(group_main_neurons, group_modulator,model='w : volt', on_pre='v_post+=w')
syn_main_mod.connect()
syn_main_mod.w = 100 * mV

# Connecting modulator group to CPG group
syn_mod_cpg = Synapses(group_modulator, group_cpg, on_pre='v_post+=10 * volt')
syn_mod_cpg.connect(i=[0, 1, 2, 3, 4, 5], j=[0, 0, 0, 1, 1, 1])

# Dopamine reward signalling
syn_reward = Synapses(aux_searching, syn_main, on_pre={'reward': 'd_post += epsilon_dopa'},
                      on_event={'reward': 'food_eaten'})
syn_reward.connect()

# Facilitate movement
aux_input_syn = Synapses(group_eff, aux_locating,  model='''''',
                   on_pre='''location_x_post += r_mov * cos(angle_turn_left_leg)
                   location_y_post += r_mov * sin(angle_turn_left_leg)
                   mov_angle_post = (mov_angle_post + angle_turn_left_leg) % 360''')
aux_input_syn.connect(i=[0, 2, 4], j=[0, 0, 0])
aux_input_syn = Synapses(group_eff, aux_locating,  model='''''',
                   on_pre='''location_x_post += r_mov * cos(angle_turn_right_leg)
                   location_y_post += r_mov * sin(angle_turn_right_leg)
                   mov_angle_post = (mov_angle_post + angle_turn_right_leg) % 360''')
aux_input_syn.connect(i=[1, 3, 5], j=[0, 0, 0])

# Sending input to main neuron group
syn_inp_main = Synapses(aux_input, group_main_neurons, model='w : volt', on_pre='v_post+=w')
syn_inp_main.connect()
syn_inp_main.w = 100 * mV

##################################################
# Simulation
##################################################
# Setting up monitors
mon_main = SpikeMonitor(group_main_neurons)
mon_eff = SpikeMonitor(group_eff)

fig, (pl1, pl2, pl3, pl4) = subplots(4, 1)

# Simulation without dopamine modulation
syn_main.mode = 0
run(simulation_duration/2, report='text')
print("Total food eaten", sum(aux_searching.eaten[:]))
pl1.set_title('STDP')
pl1.plot(mon_main.t/ms, mon_main.i, 'ok')
pl1.set_xlabel('Time, ms')
pl1.set_ylabel('Neuron index')
pl3.set_title('Motor neuron activity, STDP')
pl3.plot(mon_eff.t/ms, mon_eff.i, 'ok')
pl3.set_xlabel('Time, ms')
pl3.set_ylabel('Neuron index')

# Simulation with dopamine modulation
syn_main.mode = 1
run(simulation_duration/2, report='text')
print("Total food eaten", sum(aux_searching.eaten[:]))
pl2.set_title('Dopamine modulated STDP')
pl2.plot(mon_main.t/ms, mon_main.i, 'ok')
pl2.set_xlabel('Time, ms')
pl2.set_ylabel('Neuron index')
pl4.set_title('Motor neuron activity, dopamine modulated STDP')
pl4.plot(mon_eff.t/ms, mon_eff.i, 'ok')
pl4.set_xlabel('Time, ms')
pl4.set_ylabel('Neuron index')
tight_layout()
show()
