close all; % fecha todas as figuras
clear;     % limpa variáveis do workspace
clc;       % limpa janela de comandos

% global ARec APly;

TT = 300.0; % tempo total de aquisicao em segundos



FA_AD = 48000; % frequencia de amostragem do A/D (tipicas 8000, 11025, 22050, 44100, 48000, and 96000)
FA_DA = FA_AD; % frequencia de amostragem do D/A (tipicas 8000, 11025, 22050, 44100, 48000, and 96000)
CH_AD = 1;     % 1 mono, 2 estereo

NSAMPLES_AD = FA_AD/25; % bloco de amostras por aquisicao do A/D.
NSAMPLES_DA = FA_DA/25; % bloco de amostras para reproducao no D/A.

% plota oscilograma do sinal de entrada
subplot(221);
xh1 = (0:NSAMPLES_AD-1)/(FA_AD*0.001);
yh1 = xh1*0;
h1 = plot(xh1,yh1); hold off; grid on;
axis([[0 NSAMPLES_AD/10]/(FA_AD*0.001) -1.0 1.0]);
xlabel('Tempo (mseg.)'), ylabel('Amplitude (norm.)');
title('Audio In - Oscilograma');
set(h1,'YDataSource','yh1');
set(h1,'XDataSource','xh1');

% plota espectro do sinal de entrada
subplot(223);
yh2 = abs(fft(yh1,length(yh1)));
yh2 = yh2(1:length(yh2)/2);
yh2 = yh2/max(yh2);
xh2 = (0:(length(yh2)-1));
xh2 = xh2*FA_AD/2/length(xh2);
h2 = plot(xh2,yh2); hold off; grid on;
axis([xh2(1) FA_AD/2 -0.1 1.1])
xlabel('Frequencia (Hz)'), ylabel('Magnitude (norm.)'),
title('Audio In - Espectro de Frequências');
set(h2,'YDataSource','yh2');
set(h2,'XDataSource','xh2');

% plota oscilograma do sinal de saida
subplot(222);
xh3 = (0:NSAMPLES_DA-1)/(FA_DA*0.001);
yh3 = xh3*0;
h3 = plot(xh3,yh3); hold off; grid on;
axis([[0 NSAMPLES_DA/10]/(FA_DA*0.001) -1.0 1.0]);
xlabel('Tempo (mseg.)'), ylabel('Amplitude (norm.)');
title('Audio Out - Oscilograma');
set(h3,'YDataSource','yh3');
set(h3,'XDataSource','xh3');

% plota espectro do sinal de saida
subplot(224);
yh4 = abs(fft(yh3,length(yh3)));
yh4 = yh4(1:length(yh4)/2);
yh4 =  yh4/max(yh4);
xh4 = (0:(length(yh4)-1));
xh4 = xh4*FA_DA/2/length(xh4);
h4 = plot(xh4,yh4); hold off; grid on;
axis([xh4(1) FA_DA/2 -0.1 1.1])
xlabel('Frequencia (Hz)'), ylabel('Magnitude (norm.)'),
title('Audio Out - Espectro de Frequências');
set(h4,'YDataSource','yh4');
set(h4,'XDataSource','xh4');

% Para versoes do MATLAB anteriores a R2020a

% ARec = dsp.AudioRecorder('SampleRate',FA_AD, ... 
%     'NumChannels',CH_AD, ...
%     'BufferSizeSource','Property',...
%     'BufferSize',NSAMPLES_AD, ... 
%     'QueueDuration',0.25, ... 
%     'SamplesPerFrame', NSAMPLES_AD);
% 
% APly = dsp.AudioPlayer('SampleRate',FA_DA, ...
%     'BufferSizeSource','Property',...
%     'BufferSize',NSAMPLES_DA, ... 
%     'QueueDuration',0.25); 
%

% Para MATLAB R2020a
ARec = audioDeviceReader('SampleRate',FA_AD,'NumChannels',CH_AD,'SamplesPerFrame',NSAMPLES_AD);
APly = audioDeviceWriter('SampleRate',FA_DA,'BufferSize',NSAMPLES_DA);
% %

%**************************************************************************
%b = 1:500;
%NFILT = length(b);       % número de coeficientes do filtro utilizado na FuncaoE7
%NFILT = max(length(b));
%zi = zeros([1 NFILT-1]); % condições iniciais para o filtro da FuncaoE7
%**************************************************************************
tic;
fprintf('Start recording.\n');


% Requisitos de projeto do equalizador parametrico

fo = 20; % frequencia central em Hz
bf = 500;  % banda para projeto do rejeita banda em Hz
gdB = 6; % ganho em dB na frequencia fo

fa = 48000; % frequencia de amostragem em Hz



lbt = 500; % largura de banda de transicao para calculo da ordem do filtro
fsi = fo - bf/2 - lbt; % frequencia limite da banda de passagem inferior em Hz
fpi = fo - bf/2;       % frequencia inferior da banda de rejeicao em Hz
fps = fo + bf/2;       % frequencia superior da banda de rejeicao em Hz
fss = fo + bf/2 + lbt ; % frequencia limite da banda de passagem superior em Hz

% Janela de Blackman: ap = 0.002 dB e as = 74 dB 
% Ver formulario de consulta disponível no Moodle

% Frequencias normalizadas (w = 2*pi*f/fa)
wsi = 2*pi*fsi/fa;
wpi = 2*pi*fpi/fa;
wps = 2*pi*fps/fa;
wss = 2*pi*fss/fa;

% Fequencia de corte do filtro passa-baixas
w1 = (wpi+wsi)/2;
w2 = (wss+wps)/2;
wc = (w2-w1)/2;

% Frequencia central do passa-banda
w0 = (w2+w1)/2;

% Ordem N do filtro (ver formulario de consulta disponível no Moodle)

% Largura da banda de transicao para projeto do rejeita banda
dw = min((wpi-wsi),(wss-wps));

% N = 4*pi/dw;  % janela retangular
% N= 2*ceil(N/2); % ordem deve ser par
% wn = boxcar(N+1);

% N = 8*pi/dw;  % janela de von Hann
% N= 2*ceil(N/2); % ordem deve ser par
% wn = hanning(N+1);

% N = 8*pi/dw;  % janela de von Hann
% N= 2*ceil(N/2); % ordem deve ser par
% wn = hamming(N+1);

N = 12*pi/dw;  % janela de Blackman
N = 2*ceil(N/2); % ordem deve ser par
wn = blackman(N+1);

% Resposta do filtro ideal atrasada de N/2 amostras para ficar causal 
% Atencao para a indeterminacao em n = N/2!
% Ver formulario de consulta disponível no Moodle
n = 0:N;
hlp = sin(wc*(n+1e-8-N/2))./(n+1e-8-N/2)/pi; % passa-baixas
hbp = 2*cos(w0*(n-N/2)).*hlp; % passa-banda
hbr = -hbp;
hbr(N/2+1) =1+hbr(N/2+1); % rejeita-banda

hn = hbr + 10^(gdB/20)*hbp;

% Resposta a amostra unitaria (impulso) do filtro a ser implementado
hn = hn.*wn';

b = hn;

%NFILT = max(length(dw));
zi = zeros([1 N]); % condições iniciais para o filtro da Funcao
% zi=N-1;

while(toc < TT),
    % le dados do A/D
    yh1 = step(ARec); 
    
    % processa dados
    [yh3 zi] = Funcao(yh1, b, zi);
           
    % escreve dados no D/A  
    step(APly, yh3);

    % atualiza oscilogramas
    refreshdata(h1);
    refreshdata(h3);

    % atualiza espectro entrada
    yh2=abs(fft(yh1,length(yh1)));
    yh2=yh2(1:length(yh2)/2);
    yh2=yh2/max(yh2);
    refreshdata(h2);

    % atualiza espectro saida
    yh4=abs(fft(yh3,length(yh3)));
    yh4=yh4(1:length(yh3)/2);
    yh4=yh4/max(yh4);
    refreshdata(h4);

    % força atualização dos gráficos
    drawnow;
    
end;
fprintf('End recording.\n');

release(ARec); % fecha dispositivo de entrada de áudio
clear ARec;    % remove ARec do workspace
release(APly); % fecha dispositivo de saída de áudio
clear APly;    % remove APly do workspace

% close all; %fecha todas as figuras