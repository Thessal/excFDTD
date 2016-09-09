%cd C:\Users\Administrator\Source\Repos\excFDTD\x64\Release\single_13_500_PML
for ind=300 %0:5:900
    a=importdata(num2str(ind,'Ex_X0_00%03d.txt'));
    b=importdata(num2str(ind,'Ey_X0_00%03d.txt'));
    c=importdata(num2str(ind,'Ez_X0_00%03d.txt'));
    t=10*(sqrt(a.^2+b.^2+c.^2));
    image(t(9+60:end-8-30,9:end-8)); colorbar(); colormap('hot'); set(gca,'Ydir','normal')
    %saveas(gcf,num2str(ind,'E2_X0_00%03d.png'));
end