#!/usr/bin/env python3
"""Pass input directly to output.

https://app.assembla.com/spaces/portaudio/git/source/master/test/patest_wire.c

"""
import argparse
import queue
import sys

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)

import math

from array import array

from scipy import signal


from scipy.signal import lfilter, lfilter_zi, butter, freqz

from numpy import array, ones


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    '-i', '--input-device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-o', '--output-device', type=int_or_str,
    help='output device (numeric ID or substring)')
parser.add_argument(
    '-c', '--channels', type=int, default=1,
    help='number of channels')
parser.add_argument('--dtype', help='audio data type')
parser.add_argument('--samplerate', type=float, help='sampling rate')
parser.add_argument('--blocksize', type=int, help='block size')
parser.add_argument('--latency', type=float, help='latency in seconds')
args = parser.parse_args(remaining)




fo = 1000 # frequencia central em Hz
bf = 500  # banda para projeto do rejeita banda em Hz
gdB = 10 # ganho em dB na frequencia fo

fa = 48000 # frequencia de amostragem em Hz

lbt = 500  # largura de banda de transicao para calculo da ordem do filtro
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


N = (12*math.pi/dw)  # janela de Blackman
N = 2*math.ceil(N/2) # ordem deve ser par
#N=300
#wn = blackman(N+1)
#wn = list(range(N))
#wn = np.array(wn)
#wn = range(N)
#for i in wn:
#    wn[i] = 0.42-0.5*math.cos((2*math.pi*i)/((N+1)-1))+0.08*math.cos((4*math.pi*i)/((N+1)-1))
wn = np.blackman(N+1)
#print(wn)
# Resposta do filtro ideal atrasada de N/2 amostras para ficar causal
# Atencao para a indeterminacao em n = N/2!
# Ver formulario de consulta disponível no Moodle
hlp = list(range(N))
#hlp = range(N)
hbp = list(range(N))
hbr = list(range(N))
hn = list(range(N))
#hlp = signal.butter(N, wn, btype='low', analog=False)
for n in range(N):
    hlp[n] = math.sin(wc*(n+1e-8-N/2))/(n+1e-8-N/2)/math.pi # passa-baixas
#for n in range(0,N):
    hbp[n] = 2*math.cos(w0*(n-N/2))*hlp[n] # passa-banda
#for n in range(0,N):
    hbr[n] = hbp[n]*(-1)

#print(hlp)
#print(math.pi)
#print(N)
#print(gg)
#print(hbr)
hbr[math.ceil((N/2)+1)] = 1 + hbr[math.ceil((N/2)+1)] # rejeita-banda
gdb = 10**(gdB/20)
for n in range(1,N):
    hn[n] = hbr[n] + gdb*hbp[n]

# Resposta a amostra unitaria (impulso) do filtro a ser implementado
for n in range(0,N):
    hn[n] = hn[n]*wn[n]

#print(gdb)
#print(hn[105])


zi1 = list(range(N-1))
for n in range(1, (N-1)):
    zi1[n] = 0

#t = list(range(N))
#t[0] = 1
#zi1[1] = 0


b = np.array(hn)
zi = np.array(zi1)

#print(b)
#print(b)
#print(zi)
# Plot the frequency response.
#w, mag, phase = signal.bode(b)

#plt.figure()
#plt.semilogx(w, mag)    # Bode magnitude plot
#plt.figure()
#plt.semilogx(w, phase)  # Bode phase plot
#plt.show()

#print(hn)
#print(b[0])
#print(b[1])
#print(len(b))
#b, a = signal.butter(3, 0.05)
print(len(b))
a = 1
#b = 1
#zi = signal.lfiltic(b, a)
#zi = signal.lfilter_zi(b, 1)


def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    #print(len(indata))
    #print(len(outdata))
    global zi
    concat_list = [j for i in indata for j in i]

    i = np.array(concat_list)

    y, zf = lfilter(b, 1, i, zi=zi*1)

    zi = zf

    o = []
    for var in y:
        aux = []
        aux.append(var)
        o.append(aux)
    #print(o)
    o1 = np.array(o)
    #print(len(o1))
    #print(len(indata))

    #o1 = len(o1)/2
    #print(len(o1))
    outdata[:] = o1


try:
    with sd.Stream(device=(args.input_device, args.output_device),
                   samplerate=48000, blocksize=args.blocksize,
                   dtype=args.dtype, latency=args.latency,
                   channels=args.channels, callback=callback):
        print('#' * 80)
        print('press Return to quit')
        print('#' * 80)
        input()
except KeyboardInterrupt:
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
