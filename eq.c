#include <stdio.h>
#include <math.h>

# define M_PI 3.14159265358979323846


float min(float min1, float min2)
{
	if(min1<min2){
		return min1;
	}
	else return min2;
}


int main()
{
	// Requisitos de projeto do equalizador parametrico

	int fo = 8000; //% frequencia central em Hz
	int bf = 500;  //% banda para projeto do rejeita banda em Hz
	int gdB = 1; //% ganho em dB na frequencia fo

	int fa = 48000; //% frequencia de amostragem em Hz

	int lbt = 500; //% largura de banda de transicao para calculo da ordem do filtro
	int fsi = fo - bf/2 - lbt; //% frequencia limite da banda de passagem inferior em Hz
	int fpi = fo - bf/2;       //% frequencia inferior da banda de rejeicao em Hz
	int fps = fo + bf/2;       //% frequencia superior da banda de rejeicao em Hz
	int fss = fo + bf/2 + lbt ; //% frequencia limite da banda de passagem superior em Hz

	//% Janela de Blackman: ap = 0.002 dB e as = 74 dB 
	//% Ver formulario de consulta disponível no Moodle

	//% Frequencias normalizadas (w = 2*pi*f/fa)
	float wsi = 2*M_PI*fsi/fa;
	float wpi = 2*M_PI*fpi/fa;
	float wps = 2*M_PI*fps/fa;
	float wss = 2*M_PI*fss/fa;

	//% Fequencia de corte do filtro passa-baixas
	float w1 = (wpi+wsi)/2;
	float w2 = (wss+wps)/2;
	float wc = (w2-w1)/2;

	//% Frequencia central do passa-banda
	float w0 = (w2+w1)/2;

	//% Ordem N do filtro (ver formulario de consulta disponível no Moodle)

	//% Largura da banda de transicao para projeto do rejeita banda
	float dw = min((wpi-wsi),(wss-wps));

	//% N = 4*pi/dw;  % janela retangular
	//% N= 2*ceil(N/2); % ordem deve ser par
	//% wn = boxcar(N+1);

	//% N = 8*pi/dw;  % janela de von Hann
	//% N= 2*ceil(N/2); % ordem deve ser par
	//% wn = hanning(N+1);

	//% N = 8*pi/dw;  % janela de von Hann
	//% N= 2*ceil(N/2); % ordem deve ser par
	//% wn = hamming(N+1);

	int N = (12*M_PI)/dw;//  % janela de Blackman
	//printf("%d\n", N);
	N = 2*ceil(N/2); //% ordem deve ser par
	printf("%d\n", N);
	//wn = blackman(N+1);
	int i=0;
	double wn[N+1];
	double x1, x2;
	for(i=0;i<=N+1;i++)
	 {
		x1 = (2*M_PI*i)/((N+1)-1);
		x2 = (4*M_PI*i)/((N+1)-1);
		
		wn[i] = 0.42-0.5*cos((2*M_PI*i)/((N+1)-1))+0.08*cos((4*M_PI*i)/((N+1)-1));
		//wn[i] = cos(i);
		//wn[i]=i+0.1;
	 }
	
	//for(i=1;i<=5;i++)
	 //{
		 //printf("valor de wn(%d) e: %f\n", i, wn[i]);
	 //}
	//% Resposta do filtro ideal atrasada de N/2 amostras para ficar causal 
	//% Atencao para a indeterminacao em n = N/2!
	//% Ver formulario de consulta disponível no Moodle
	int n=0;
	double hlp[N+1], hbp[N+1], hbr[N+1], hn[N+1];
	for(n=0;n<N;n++)
	{
		//n = 0:N;
		hlp[n] = sin(wc*(n+1e-8-N/2))/(n+1e-8-N/2)/M_PI; //% passa-baixas
		hbp[n] = 2*cos(w0*(n-N/2))*hlp[n]; //% passa-banda
		hbr[n] = hbp[n]*(-1);
		
	}
	printf("O valor de HN eh: %f\n\n", hbr[N/2+1]);
	
		hbr[N/2+1] =1+hbr[N/2+1]; //% rejeita-banda
	
	printf("O valor de HN eh: %f\n\n", pow(10, (gdB/20)));
	
	for(n=0;n<N;n++)
	{
		hn[n] = hbr[n] + pow(10, (gdB/20))*hbp[n];
	}

	//% Resposta a amostra unitaria (impulso) do filtro a ser implementado
	int j;
	for(j=0;j<N;j++)
	{
		hn[j] = hn[j]*wn[j];
		
	}
	
	
	
	return 0;
}
