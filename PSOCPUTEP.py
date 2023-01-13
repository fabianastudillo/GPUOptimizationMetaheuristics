import numpy as np
import cupy as cp 
import random as rand
import matplotlib.pyplot as plt
import pandapower as pp 
from numpy import array
import numpy as np
# libreria para GPU 
from numpy import matlib
from timeit import default_timer as timer
TEPtotal = 0
#DATOS IMPORTANTES
LineasIniciales = np.array([1,0,1,1,0,1,1,0,0,0,1,0,0,0,0])
PN = 10;        #Numero de particulas
dim = 15;       #La dimension es el numero de posibles vias que se pueden crear en la topologia
Xmax = 5;       #Numero maximo de lineas que se pueden agregar entre nodos. 
Xmin = LineasIniciales    #Topologia incial del problema

chi= 0.729;
c1 = 2.05;
c2 = 2.05;

#Creacion Xmax
Xmax = Xmax;
Xmax = np.matlib.repmat(Xmax,dim,PN)
#Creacion Xmin  
Xmin= np.matlib.repmat(Xmin, PN,1)
Xmin = np.transpose(Xmin)
#Valocidad maxima
vmax = 0.5*(Xmax-Xmin)


def CreacionInicial(LineasIniciales, PN, dim, Xmax, Xmin):
    import pandapower as pp 
    from numpy import array
    import numpy as np
    
    #OPCION 2 DE CREACION INCIAL
    #SE ASIGNA UN NUMERO DE LINEAS PARA CADA DERECHO DE VIA DE FORMA ALEATORIA, SIGUIENTE LOS SIGUIENTES PARAMETROS.
    
    swarm=np.round(Xmin+(Xmax-Xmin)*(np.random.rand(dim,PN)));
    print(swarm)
    for i in range(0,PN):
        alpha=(0.6+0.25*np.random.rand());  
        indices=np.random.randint( np.round(alpha*dim),size=15);
        indices = np.transpose(indices);
        swarm[indices,i]=Xmin[indices,i];
    
    print("Posicion inicial del enjambre 100 particulas")
        
    for j in range(0,PN):
        print(swarm[:,j])
        
    return swarm
#OBTENCION DE LAS POSICIONES DE LAS PARTICULAS 
swarm2 = CreacionInicial(LineasIniciales,PN, dim, Xmax, Xmin)
#INICIALIZACION DE LA MEJOR POSICION
bestpos = swarm2; 
#Velocidad inicial
vel = (2*np.random.rand(dim, PN))*(Xmax/2)-(Xmax/2);

def EvaluacionParticulas(swarm, PN):
    import pandapower as pp 
    costo =np.array([40,38,60,20,68,20,40,31,30,59,20,48,63,30,61])*1000
    coste = np.empty((PN,1))
    for i in range (0,PN):
        net = pp.create_empty_network()
        penalizacion = 0
        min_vm_pu = 0.95
        max_vm_pu = 1.05
    
    # TOPOLOGIA PARA UN SISTEMA GARVER
    
        bus1 = pp.create_bus(net, vn_kv = 110,geodata=(10,20), min_vm_pu = min_vm_pu, max_vm_pu = max_vm_pu)
        bus2 = pp.create_bus(net, vn_kv = 110, geodata=(5,15), min_vm_pu = min_vm_pu, max_vm_pu = max_vm_pu)
        bus3 = pp.create_bus(net, vn_kv = 110, geodata=(1,18), min_vm_pu = min_vm_pu, max_vm_pu = max_vm_pu)
        bus4 = pp.create_bus(net, vn_kv = 110, geodata=(10,10), min_vm_pu = min_vm_pu, max_vm_pu = max_vm_pu)
        bus5 = pp.create_bus(net, vn_kv = 110, geodata=(5,20), min_vm_pu = min_vm_pu, max_vm_pu = max_vm_pu)
        bus6 = pp.create_bus(net, vn_kv = 110, geodata=(1,10), min_vm_pu = min_vm_pu, max_vm_pu = max_vm_pu)
        
        vias = [[bus1,bus1,bus1,bus1,bus1,bus2,bus2,bus2,bus2,bus3,bus3,bus3,bus4,bus4,bus5],
                [bus2,bus3,bus4,bus5,bus6,bus3,bus4,bus5,bus6,bus4,bus5,bus6,bus5,bus6,bus6]]
        

        
        #creacion de las lineas a 110 Kw con bucle
        #se asigna el derecho de via correpondiente, dependiendo si existe o no la via.  
        
        cont = 0;
        for j in swarm[:,i]:
            if j != 0:
                for k in range (np.int(j)):
                    exec('l{} = pp.create_line(net, vias[0][cont],vias[1][cont],parallel=4,max_loading_percent = 65, length_km = 1., std_type = "149-AL1/24-ST1A 110.0")'.format(cont))
            cont = cont +1; 
        print("Se crearon las lineas AC")
        
        
            #Creacion de los generadores 
        
        g1 = pp.create_gen(net,bus1, p_mw = 50, min_p_mw =0, max_p_mw =60, controllable = True, slack= True)
        g2 = pp.create_gen(net,bus3, p_mw = 165, min_p_mw =0, max_p_mw =180 , controllable = True)
        g3 = pp.create_gen(net,bus6, p_mw = 545, min_p_mw =0, max_p_mw =600 , controllable = True)
        
        #Creacion de las cargas 
        
        pp.create_load(net, bus1, p_mw = 80)  
        pp.create_load(net, bus2, p_mw = 240)  
        pp.create_load(net, bus3, p_mw = 40)  
        pp.create_load(net, bus4, p_mw = 160)  
        pp.create_load(net, bus5, p_mw = 240)
        
        
        #Creacion de costos por generador en mw/h 
        
        pp.create_poly_cost(net, element = g1, et = "gen", cp1_eur_per_mw = 40)
        pp.create_poly_cost(net, element = g2, et = "gen", cp1_eur_per_mw = 20)
        pp.create_poly_cost(net, element = g3, et = "gen", cp1_eur_per_mw = 20)
        
        #Coste por derecho de via 
        costo =np.array([40,38,60,20,68,20,40,31,30,59,20,48,63,30,61])*1000
        #imprime la topologia actual        
        print(swarm[:,i])
        
    
        # SE CORRE EL SOLVER DE PUNTO INTERIOR AC 
        #pp.runopp(net, verbose=True) #imrprime todo los resultados
        print("Comienza solver")
        pp.runopp(net, numba = True)     
        #Penalizacion
        from pandapower.optimal_powerflow import a
        penalizacion = a;
        print(penalizacion)
        #Guarda los costes en un array
        a= net.res_cost + np.sum(swarm[:,i]*costo) + penalizacion;
        coste[i]= a; 
#
#        plot.simple_plot(net)
        print("Costo de la topologia")
        print(np.sum(swarm[:,i]*costo))
    return(coste)

#OBTENCION DEL COSTE  DE LAS TOPOLOGIAS
fswarm = EvaluacionParticulas(swarm2, PN)
fbestpos = fswarm;   #Inicializacion de los costos de la mejor posicion
#ACTUALIZACION DE INDICES DE LAS MEJORES PARTICULAS

fxopt = np.min(fbestpos);
ifxopt = np.argmin(fbestpos); 

#SE EXTRAE LA TOPOLOGIA DE LA MEJOR POSICION CON EL INDICE DEL MEJOR COSTE
xopt =  bestpos[:,ifxopt]

#Banderas de parada e iteracion. 
success =0;         
iteracion = 0;
fevalcount = 0; 
STOP= 0;

MaxIt = 10; # Numero maximo de iteraciones
#Array para almacenamietno de los costes de las mejores topologias. 
FOPT = np.zeros((MaxIt+1,1))

#LAZO DE EVOLUCION DEL ENJAMBRE 

while STOP == 0:     
    #ACTUALIZACION DE VELOCIDAD 
    #Ecuacion de movimiento del enjambre
    
    for i in range(0,PN):
        #Variables randomicas de PSO
        R1 = np.random.rand(dim,1)
        R2 = np.random.rand(dim,1)
        R1 = np.transpose(R1)
        R2 = np.transpose(R2)
        #Calculo de la velocidad dependiendo de la mejor posicion 
        start = timer()
        # G = chi*(vel[:,i] + c1*R1*(bestpos[:,i]-swarm2[:,i])+c2*R2*(bestpos[:,ifxopt]-swarm2[:,i]))
        G = chi*(vel[:,i] + np.multiply(R1,(bestpos[:,i]-swarm2[:,i]))*c1+np.multiply(R2,(bestpos[:,ifxopt]-swarm2[:,i]))*c2)
        TEPtime = timer() - start
        TEPtotal = TEPtotal+TEPtime
        print(TEPtotal)
        #Actualizacion del valor de velocidad
        vel[:,i] = G;
        
    #RESTRICCION DE VELOCIDADADES 
    for i in range(0,dim):
        for k in range(1,PN): 
            if vel[i,k] > vmax[i,k]:
                vel[i,k] = vmax[i,k];
            elif vel[i,k] < -vmax[i,k]:
                vel[i,k] = -vmax[i,k];
                
    #Actualizacion del enjambre 
    swarm2 = np.round(swarm2 + vel);
    
    #RESTRICCION DEL ENJAMBRE 
    
    for i in range(0,PN):
        for k in range(0,dim): 
            if swarm2[k,i] > Xmax[k,i]:
                swarm2[k,i] = Xmax[k,i];
            elif swarm2[k,i] < Xmin[k,i]:
                swarm2[k,i] = Xmin[k,i];
                
    #Evaluar particulas Actualizadas    
    fswarm = EvaluacionParticulas(swarm2, PN)
    
    if np.min(fswarm) < np.min(fbestpos):
        bestpos = swarm2;
        fbestpos = fswarm; 
        
    #ACTUALIZACION DE INDICES DE LAS MEJORES PARTICULAS

    fxopt = np.min(fbestpos);
    ifxopt = np.argmin(fbestpos); 

    #SE EXTRAE LA TOPOLOGIA DE LA MEJOR POSICION CON EL INDICE DEL MEJOR COSTE
    xopt =  bestpos[:,ifxopt]
    
    
    if iteracion >= MaxIt: 
        STOP = 1;
        

    #Almacenamiento decostes de las mejores topologias
    FOPT[iteracion] = fxopt;
    iteracion = iteracion +1;
    print("Vector de Costos de la mejor topologia")
    print(FOPT)
    print("Mejor Topologia")
    print(xopt)
        
print("tiempo PSO en CPU %f seconds." % TEPtotal)




 








