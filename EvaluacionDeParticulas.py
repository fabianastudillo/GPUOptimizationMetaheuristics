
#Creacion inicial 

import pandapower as pp 
import time
from time import perf_counter 
from numpy import array
import numpy as np
import sys
net = pp.create_empty_network()

min_vm_pu = 0.95
max_vm_pu = 1.05

# TOPOLOGIA PARA UN SISTEMA GARVER

bus1 = pp.create_bus(net, vn_kv = 110,geodata=(10,20), min_vm_pu = min_vm_pu, max_vm_pu = max_vm_pu)
bus2 = pp.create_bus(net, vn_kv = 110, geodata=(5,15), min_vm_pu = min_vm_pu, max_vm_pu = max_vm_pu)
bus3 = pp.create_bus(net, vn_kv = 110, geodata=(1,18), min_vm_pu = min_vm_pu, max_vm_pu = max_vm_pu)
bus4 = pp.create_bus(net, vn_kv = 110, geodata=(10,10), min_vm_pu = min_vm_pu, max_vm_pu = max_vm_pu)
bus5 = pp.create_bus(net, vn_kv = 110, geodata=(5,20), min_vm_pu = min_vm_pu, max_vm_pu = max_vm_pu)
bus6 = pp.create_bus(net, vn_kv = 110, geodata=(1,10), min_vm_pu = min_vm_pu, max_vm_pu = max_vm_pu)

LineasIniciales = np.array([2,0,1,3,0,3,1,0,0,0,1,0,0,1,0])


vias = [[bus1,bus1,bus1,bus1,bus1,bus2,bus2,bus2,bus2,bus3,bus3,bus3,bus4,bus4,bus5],
        [bus2,bus3,bus4,bus5,bus6,bus3,bus4,bus5,bus6,bus4,bus5,bus6,bus5,bus6,bus6]]


costo =np.array([40,38,60,20,68,20,40,31,30,59,20,48,63,30,61])*1000

#creacion de las lineas a 110 Kw con bucle
#se asigna el derecho de via correpondiente, dependiendo si existe o no la via.  

cont = 0;
for i in LineasIniciales:
    if i != 0:
        for k in range (i):
            exec('l{} = pp.create_line(net, vias[0][cont],vias[1][cont],parallel=3,max_loading_percent = 50, length_km = 1., std_type = "149-AL1/24-ST1A 110.0")'.format(cont))
    cont = cont +1; 
print("Se crearon las lineas AC")

    
#Creacion de los generadores 

g1 = pp.create_gen(net,bus1, p_mw = 1000, min_p_mw =0, max_p_mw =450 , controllable = True, slack= True)
g2 = pp.create_gen(net,bus3, p_mw = 600, min_p_mw =0, max_p_mw =450 , controllable = True)
g3 = pp.create_gen(net,bus6, p_mw = 600, min_p_mw =0, max_p_mw =450 , controllable = True)

#Creacion de las cargas 

pp.create_load(net, bus1, p_mw = 80)  
pp.create_load(net, bus2, p_mw = 240)  
pp.create_load(net, bus3, p_mw = 400)  
pp.create_load(net, bus4, p_mw = 300)  
pp.create_load(net, bus5, p_mw = 240)


#Creacion de costos por generador en mw/h 

pp.create_poly_cost(net, element = g1, et = "gen", cp1_eur_per_mw = 40)
pp.create_poly_cost(net, element = g2, et = "gen", cp1_eur_per_mw = 20)
pp.create_poly_cost(net, element = g3, et = "gen", cp1_eur_per_mw = 20)


#Coste por derecho de via 


#pp.runopp(net, verbose=True) 
start = time.time()
pp.runopp(net, numba = True, check_connectivity = True)
end = time.time()
print("Tiempo que toma el flujo optimo = %s" % (end - start))
from pandapower.optimal_powerflow import a    #Se importa la bandera del script optimal_powerflow.py 
pena = a;
#Final de medicion de tiempo 
#toc=  perf_counter()
#testimado = toc-tic

print("potencia suministrada por cada generador")
print(net.res_gen)
print("Costo de la topologia")
print(net.res_cost + np.sum(LineasIniciales*costo))
#print("tiempo estimado")
#print(testimado)
#print("informacion de lineas")
#print(net.res_dcline)
print("informacion de lineas")
print(net.res_line)
print("penalizacion")
print(pena)

import pandapower.plotting as plot

plot.simple_plot(net)
