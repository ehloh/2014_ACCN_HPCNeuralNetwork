clear all; clc; close all hidden



% Other setup 
where='/Users/EleanorL/Dropbox/SANDISK/12_ACCN/1_Project/';



%% Bargraph: Population means + SEs

% Background
b_popmeans=[4.4986979166666687, 4.2256944444444446, 2.2975260416666692];
b_popses=[0.027766755354343885, 0.028626478216125389, 0.020367640816556145];

% Active
a_popmeans=[5.0753038194444384, 6.4902343749999947, 3.103081597222217];
a_popses=[0.054140860655546692, 0.02929558092563778, 0.021247051264974712];

% Theta inputs 
% t_popmeans=[4.5716145833333321, 6.1369357638888964, 2.4498697916666683];
% t_popses=[0.029337163655824568, 0.029134386466835565, 0.022278893582080319];


% Results 1
% t_popmeans=[4.5839843749999991, 6.9320746527777786, 2.522786458333333];
% t_popses=[0.028932265296830174, 0.034438227673952305, 0.023212118413784268];

t_popmeans=[4.5544704861111143, 7.4572482638888911, 2.5284288194444446]
t_popses=[0.027937939291287763, 0.033408496242164867, 0.023306859475885185]


%%

% Collate overall data : row=population, col= condition
popmeans=[b_popmeans' a_popmeans' t_popmeans' ];
popses=[b_popses' a_popses' t_popses' ];

% Settings
FontSize=18;
yrange_hz=[2 8];
% fontname='PT Sans Caption';  % pt serif (caption) ,san serif , pt sans,trebuchet ms
% fontname='Cambria';
fontname='Arial';

barwitherr(popses*2, popmeans)
legend(gca,{'Background';'Active'; 'Theta'},'Location', 'Best','FontSize', FontSize, 'FontName', fontname); %

set(gca,'XTick',[1 2 3], 'FontName', fontname);
set(gca,'XTickLabel', {'CA3';'CA1'; 'Amygdala'}, 'FontSize', FontSize, 'FontName', fontname); %
axis([0 4  yrange_hz(1) yrange_hz(2)])
ylabel('Mean firing rate (Hz)','FontSize', FontSize, 'FontName', fontname); 
set(gcf,'Color','w')
