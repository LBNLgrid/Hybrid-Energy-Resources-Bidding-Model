import numpy as np
import yaml
def process_input_data(datapath):
    hh = datapath + "config.yml"
    with open(hh) as stream:
        input_data = yaml.safe_load(stream)
    esr=input_data['Esr']
    #esr.init = input_data['init'].values[2]
    if 'init' in esr:
        pass
    else:
        raise ValueError('provide initial value of ESR (in MW) ')

    if 'max' in esr:
        pass
    else:
        raise ValueError('provide Maximum limit of ESR (in MW) ')
    if 'min' in esr:
        pass
    else:
        raise ValueError('provide Minimum limit of ESR (in MW) ')

    if 'effCrg' in esr:
        pass
    else:
        raise ValueError('provide charging efficiency of ESR in (0,1)')

    if 'effDis' in esr:
        pass
    else:
        raise ValueError('provide discharging efficiency of ESR in (0,1)')

    if 'opCost' in esr:
        pass
    else:
        raise ValueError('provide operating cost of ESR')

    if 'power' in esr:
        pass
    else:
        raise ValueError('provide charging/discharging rate of ESR (in MW)')

    #print (esr)
    gen = input_data['Gen']

    if 'opCost' in gen:
        pass
    else:
        raise ValueError('provide operating cost of generator ')

    if 'max' in gen:
        pass
    else:
        raise ValueError('provide maximum capacity of generator')

    POI =input_data['Hybrid']['POI']
    if np.isnan(POI):
        raise ValueError('provide point of interconnection power capacity')

    Hyb_name = input_data['Hybrid']['Name']
    if np.isnan(POI):
        raise ValueError('provide Hybrid name')
    return Hyb_name,POI, esr, gen