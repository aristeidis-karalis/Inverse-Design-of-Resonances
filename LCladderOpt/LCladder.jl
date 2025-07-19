using NLsolve
include("ABCD.jl")

function Series(s,L,iC)
    # Using the exp(st) convention
    Z = L*s+iC/s
    M  = [1 Z; 0  1]
#    dM_ds = [0 L-iC/s^2; 0 0]
    dM_dL = [0 s; 0 0]
    dM_diC = [0 1/s; 0 0]
    return M, dM_dL, dM_diC
end    

function Shunt(s,C,iL)
    # Using the exp(st) convention
    Y = C*s+iL/s
    M  = [1 0; Y  1]
#    dM_ds = [0 C-iL/s^2; 0 0]
    dM_dC = [0 0; s 0]
    dM_diL = [0 0; 1/s 0]
    return M, dM_dC, dM_diL
end    


function LCladder_M_dM(s,l,c; derivs=false)
#   Calculates ABCD matrix for a ladder of LC series and shunt branches.
    N = length(l)

    ## Calculate M=ABCD matrix.
    dM_dl = zeros(ComplexF64, 2,2,N)
    dM_dc = zeros(ComplexF64, 2,2,N)
    if derivs; Ms = zeros(ComplexF64, 2,2,N); end

    ## Propagating through the layers.
    jj = 1
    M = [1 0; 0 1]
    while jj<=N
        if isodd(jj)
            Mj, dM_dlj, dM_dcj = Series(s,l[jj],c[jj])
        else
            Mj, dM_dcj, dM_dlj = Shunt(s,c[jj],l[jj])
        end
        if derivs
            Ms[:,:,jj] = Mj
            dM_dl[:,:,jj] = M * dM_dlj
            dM_dc[:,:,jj] = M * dM_dcj
        end
        M = M * Mj
        jj += 1
    end

    ## Calculate derivatives
    if derivs
        jj = N
        Mr = [1 0; 0 1]
        while jj >= 1
            dM_dl[:,:,jj] *= Mr
            dM_dc[:,:,jj] *= Mr
            Mr = Ms[:,:,jj] * Mr
            jj -= 1
        end
    end

    return M, dM_dl, dM_dc
end

function LCladder_S(ω,l,c,RG,RL)

    Nω = length(ω)
    #M = zeros(ComplexF64, 2,2,Nω)
    S = zeros(ComplexF64, 2,2,Nω)
    for i=1:Nω
        Mi, _, _ = LCladder_M_dM(-im*ω[i],l,c)
        #M[:,:,i] = Mi
        S[:,:,i] = M2S(Mi,[],RG,RL)
    end

    return S
end

function LCladder_res(ωg,l,c,RG,RL)

    function LCladder_res!(F,s)
        M, _, _ = LCladder_M_dM(s[1],l,c)
        F[1] = M_res(M,RG,RL)
    end
        
    s = [0.0-im*ωg]
    sol = nlsolve(LCladder_res!, s)

    s = sol.zero[1]
    M, _, _ = LCladder_M_dM(s,l,c)

    ARL = M[1,1]*RL
    B = M[1,2]
    CRGL = M[2,1]*RG*RL
    DRG = M[2,2]*RG

    σ = 2*sqrt(RG*RL)/(ARL+B-CRGL-DRG) # = 1/T21 = S21/S11

    return im*s, σ, sol
end