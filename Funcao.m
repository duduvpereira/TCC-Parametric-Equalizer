% Aplicacoes de Processamento Digital de Sinais - 4456S-04
% Experiência E7: Projeto e implementação de filtros FIR
% Prof. Denis Fernandes 
% Ultima atualizacao: 30/04/2019

function [y, zf] = Funcao(x, b, zi)
% x - sinal a ser filtrado
% zi - condicoes iniciais do filtro
% zi - condicoes finais do filtro
% y - sinal filtrado


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%B Pratica
% y = x*0;
% 
%  for n = 1:length(x)
%     y(n)=x(n)*b(1) +b(2:end)*zi';
%     zi=[x(n),zi(1:end-1)];
%  end
%  zf=zi;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Filtragem
%********************************************

[y zf] = filter(b,1,x,zi);

%********************************************

end