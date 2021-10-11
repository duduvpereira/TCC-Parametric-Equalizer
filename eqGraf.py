
#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
#import control

#import warnings
#warnings.filterwarnings('ignore')

import math

#from pylab import *

# Requisitos de projeto do equalizador parametrico

fo = 8000 # frequencia central em Hz
bf = 500  # banda para projeto do rejeita banda em Hz
gdB = 10 # ganho em dB na frequencia fo

fa = 48000 # frequencia de amostragem em Hz

lbt = 500 # largura de banda de transicao para calculo da ordem do filtro
fsi = fo - bf/2 - lbt # frequencia limite da banda de passagem inferior em Hz
fpi = fo - bf/2       # frequencia inferior da banda de rejeicao em Hz
fps = fo + bf/2       # frequencia superior da banda de rejeicao em Hz
fss = fo + bf/2 + lbt  # frequencia limite da banda de passagem superior em Hz

# Janela de Blackman: ap = 0.002 dB e as = 74 dB
# Ver formulario de consulta disponível no Moodle

# Frequencias normalizadas (w = 2*pi*f/fa)
wsi = 2*math.pi*fsi/fa
wpi = 2*math.pi*fpi/fa
wps = 2*math.pi*fps/fa
wss = 2*math.pi*fss/fa

# Fequencia de corte do filtro passa-baixas
w1 = (wpi+wsi)/2
w2 = (wss+wps)/2
wc = (w2-w1)/2

# Frequencia central do passa-banda
w0 = (w2+w1)/2

# Ordem N do filtro (ver formulario de consulta disponível no Moodle)

# Largura da banda de transicao para projeto do rejeita banda
dw = min((wpi-wsi),(wss-wps))

# N = 4*pi/dw  # janela retangular
# N= 2*ceil(N/2) # ordem deve ser par
# wn = boxcar(N+1)

# N = 8*pi/dw  # janela de von Hann
# N= 2*ceil(N/2) # ordem deve ser par
# wn = hanning(N+1)

# N = 8*pi/dw  # janela de von Hann
# N= 2*ceil(N/2) # ordem deve ser par
# wn = hamming(N+1)


deeee = (12*math.pi/dw)  # janela de Blackman
N = 2*math.ceil(deeee/2) # ordem deve ser par
#wn = blackman(N+1)
wn = list(range(N))
for i in range(1,N):
    wn[i] = 0.42-0.5*math.cos((2*math.pi*i)/((N+1)-1))+0.08*math.cos((4*math.pi*i)/((N+1)-1))

# Resposta do filtro ideal atrasada de N/2 amostras para ficar causal
# Atencao para a indeterminacao em n = N/2!
# Ver formulario de consulta disponível no Moodle
hlp = list(range(N))
hbp = list(range(N))
hbr = list(range(N))
hn = list(range(N))
for n in range(1,N):
    hlp[n] = math.sin(wc*(n+1e-8-N/2))/(n+1e-8-N/2)/math.pi # passa-baixas
for n in range(1,N):
    hbp[n] = 2*math.cos(w0*(n-N/2))*hlp[n] # passa-banda
for n in range(1,N):
    hbr[n] = hbp[n]*(-1)

gg = (N/2)+1
gg=math.ceil(gg)
#print(hlp)
print(N)
print(gg)
#print(hbr)
hbr[gg] = 1 + hbr[gg] # rejeita-banda
gdb = 10^(math.ceil(gdB/20))
for n in range(1,N):
    hn[n] = hbr[n] + gdb*hbp[n]

# Resposta a amostra unitaria (impulso) do filtro a ser implementado
for n in range(1,N):
    hn[n] = hn[n]*wn[n]

print(hn)


#mag,phase,omega = control.bode(hn)
# Visualizacao da resposta do filtro
#fvtool(hn, 'fs',fa)
