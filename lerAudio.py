#!/usr/bin/env python3
"""Plot the live microphone signal(s) with matplotlib.

Matplotlib and NumPy have to be installed.

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


from scipy.signal import lfilter, lfilter_zi, butter

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
    'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
    help='input channels to plot (default: the first)')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-w', '--window', type=float, default=200, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
parser.add_argument(
    '-i', '--interval', type=float, default=30,
    help='minimum time between plot updates (default: %(default)s ms)')
parser.add_argument(
    '-b', '--blocksize', type=int, help='block size (in samples)')
parser.add_argument(
    '-r', '--samplerate', type=float, help='sampling rate of audio device')
parser.add_argument(
    '-n', '--downsample', type=int, default=10, metavar='N',
    help='display every Nth sample (default: %(default)s)')
args = parser.parse_args(remaining)
if any(c < 1 for c in args.channels):
    parser.error('argument CHANNEL: must be >= 1')
mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1
q = queue.Queue()


fo = 1000 # frequencia central em Hz
bf = 500  # banda para projeto do rejeita banda em Hz
gdB = -50 # ganho em dB na frequencia fo

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

#print(hlp)
print(N)
#print(gg)
#print(hbr)
hbr[math.ceil((N/2)+1)] = 1 + hbr[math.ceil((N/2)+1)] # rejeita-banda
gdb = 10^(math.ceil(gdB/20))
for n in range(1,N):
    hn[n] = hbr[n] + gdb*hbp[n]

# Resposta a amostra unitaria (impulso) do filtro a ser implementado
for n in range(1,N):
    hn[n] = hn[n]*wn[n]

zi1 = list(range(N-1))
for n in range(1,(N-1)):
    zi1[n] = 0

b = np.array(hn)
zi = np.array(zi1)

print(b)


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)

    # Fancy indexing with mapping creates a (necessary!) copy:
    #for n in range(1,5000):
    #    indata = indata*-1
    #print(len(indata))
    #print(indata)
    #indata = np.array(indata)
    #zi = signal.lfilter_zi(b, 1)

    #z, _ = signal.lfilter(b, 1, indata, zi=zi*indata[0])
    #zi = lfilter_zi(b, 1)
    #print(zi)
    #print(zi)
    #print(indata)
    concat_list = [j for i in indata for j in i]

    i = np.array(concat_list)

    #print(len(concat_list))
    #print(len(i))
    #print(len(b))
    #print(len(zi))
    #print(i)
    #print(b)
    #print(indata[0])
    #y, zo = lfilter(b, 1, ones(10), zi=zi)
    y, zf = lfilter(b, 1, i, zi=zi*i[0])
    #print(y)
    #print(type(y))
    #print(len(y))
    o = []
    for var in y:
        aux = []
        aux.append(var)
        o.append(aux)
    #print(o)
    o1 = np.array(o)
    #print(o1)
    #print(type(indata))
    #for n in range(1,N):
    #    y[n]=indata[n]*b[n]
    #print(type(indata))

    q.put(o1[::args.downsample, mapping])

def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    outdata[:] = indata

def update_plot(frame):
    """This is called by matplotlib for each plot update.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global plotdata
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])
    return lines

try:
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = device_info['default_samplerate']

    length = int(args.window * args.samplerate / (1000 * args.downsample))
    plotdata = np.zeros((length, len(args.channels)))

    fig, ax = plt.subplots()
    lines = ax.plot(plotdata)
    if len(args.channels) > 1:
        ax.legend(['channel {}'.format(c) for c in args.channels],
                  loc='lower left', ncol=len(args.channels))
    ax.axis((0, len(plotdata), -1, 1))
    ax.set_yticks([0])
    ax.yaxis.grid(True)
    ax.tick_params(bottom=False, top=False, labelbottom=False,
                   right=False, left=False, labelleft=False)
    fig.tight_layout(pad=0)

    stream = sd.InputStream(
        device=args.device, channels=max(args.channels),
        samplerate=48000, callback=audio_callback)

    #stream = sd.Stream(channels=2, callback=callback)

    ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=True)
    with stream:
        plt.show()


except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
