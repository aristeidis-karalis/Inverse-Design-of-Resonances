using LinearAlgebra

function T2S(T; M1=1,M2=1)

    M,N = size(T)
    if M!=2M1 || N!=2M2; error("The number of ports is wrong."); end
    
    T11 = T[1:M1,1:M2]
    T12 = T[1:M1,M2+1:N]
    T21 = T[M1+1:M,1:M2]
    T22 = T[M1+1:M,M2+1:N]
    
    S21 = T11\LinearAlgebra.I(M1)
    S22 = -T11\T12; # = -S21*T12;
    S11 = T21*S21;
    S12 = T22+T21*S22; # = T22-T21*S21*T12;
    
    S = [S11 S12; S21 S22]
end