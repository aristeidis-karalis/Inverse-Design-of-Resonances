function M2S(M,dM,RG,RL)
    #   Convert from ABCD to S matrix
    ARL = M[1,1]*RL
    B = M[1,2]
    CRGL = M[2,1]*RG*RL
    DRG = M[2,2]*RG

    Δ = ARL + B + CRGL + DRG
    S11 = ARL + B - CRGL - DRG
    S21 = 2*sqrt(RG*RL)
    S12 = S21*det(M)
    S22 = - ARL + B - CRGL + DRG


    if !isempty(dM)
        N = size(dM,3)
        dS = zeros(eltype(M), 2,2,N)

        dARL = dM[1,1,:]*RL
        dB = dM[1,2,:]
        dCRGL = dM[2,1,:]*RG*RL
        dDRG = dM[2,2,:]*RG

        dS11 = ((dARL+dB)*(CRGL+DRG)-(ARL+B)*(dCRGL+dDRG))*2
        dS21 = - S21 * (dARL + dB + dCRGL + dDRG)
        dS12 = dS21 # fix for non-reciprocity
        dS22 = ((dDRG+dB)*(CRGL+ARL)-(DRG+B)*(dCRGL+dARL))*2
        for i=1:N; dS[:,:,i] .= [dS11[i] dS12[i]; dS21[i] dS22[i]]./Δ^2; end
    end

    S = [S11 S12; S21 S22]./Δ

    if isempty(dM); return S; else; return (S, dS); end
end

function M_res(M,RG,RL)
    ARL = M[1,1]*RL
    B = M[1,2]
    CRGL = M[2,1]*RG*RL
    DRG = M[2,2]*RG
    
    Δ = ARL + B + CRGL + DRG
    return Δ
end