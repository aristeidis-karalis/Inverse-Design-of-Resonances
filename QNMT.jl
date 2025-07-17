# Sbar calculation
function calc_Sbar(freq, poles, σ, P::Int)
    # freq is scalar frequency
    # poles and σ are (column) vectors
    # P = number of ports
    
    N = length(poles)
    sPoles = transpose(1im*poles)
    D = transpose([ones(N) σ])
    M = (D'*D)./(sPoles' .+ sPoles)
    K = conj(D/M); # This is K without C.

    # Calculate Sb=\bar{S}, its derivative and its pole residues.
    Sb = zeros(ComplexF64, P,P);
    dSb = zeros(ComplexF64, P,P);
    for p = 1:P
        for q = 1:P
            rSb_pq = D[p,:].*K[q,:]
            rSb_pq = conj(rSb_pq')
            Sb[p,q] = (p==q) + sum(rSb_pq./(1im*freq .- sPoles))
            dSb[p,q] = -1im* sum(rSb_pq./(1im*freq .- sPoles).^2)
        end
    end
    return Sb, dSb
end

