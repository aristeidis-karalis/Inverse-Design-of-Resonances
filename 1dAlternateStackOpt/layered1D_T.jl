function layered_iso_T_dT(w, kp, e, m, d, bc, pol; derivs=false)
# Calculates Transmission matrix for 1d z-stack of layers and its derivative w.r.t. permittivities and thicknesses.
#=    
     INPUTS:
     -	w: frequency
     -	kp: paralel wavevector (in the xy plane of symmetry), so normal incidence is kp=0
     -	e: vector of relative permittivities of ISOTROPIC layers
     -	m: vector of relative permeabilities of ISOTROPIC layers
     -	d: vector of thicknesses of layers (Inf or 0 for first and/or last, if port at Infinity)
     -	bc: boundary conditions at beginning and end of stack in the form 'ab', where 'a','b' can be
           ~‘e’ for PEC
           ~‘m’ for PMC
           ~‘o’ for open when you are looking for guided modes
           ~‘l’ for open when you are looking for leaky modes
     -	pol: polarization = ‘e’ for TE (E only in xy) or ‘m’ for TM (H only in xy)
     
     OUTPUTS: 
     - T: complex Transfer matrix (always 2x2)
     - dT_de: Jacobian of Transfer matrix with e (always 2x2xN)
     - dT_dd: Jacobian of Transfer matrix with d (always 2x2xN)
     - X: complex vector of layer admittances for TE (kz/w/m) or impedances for TM (kz/w/e)
     - kz: complex vector of layer z-wavevector
     
     NOTES: 
     - Input quantities should always be normalized as follows:
           ~d to some length-scale L 
           ~k to 2π/L
           ~w to 2π*c/L
           ~f to c/L
           so a phase is always 2π*k*d.
     - Convention exp(-iwt) is used, so forward propagating is exp(ikz).
     - Reflection coefficient (Snn) is defined as Snn=En_ref/En_inc for TE, 
           but Snn=Hn_ref/Hn_inc for TM (to make equations dual), 
           so at normal incidence (k=0) there is (unfortunately) a minus-sign difference!
    
     KNOWS ISSUES: 
     - Does not work for e=0 or m=0, but works for X=kz=0 and/or d=0.
     
     EXAMPLE:
     TE t/r through a 3mm dielectric (ε=5) layer at 10GHz and 60deg incidence:
     T,dT_de,dT_dd,X,kz = layered_iso_T(1,1*sin(π/3),[1,5,1],[1,1,1],[Inf,0.1,Inf],'oo','e')
     Since we set w=1, all thicknesses should be normalized to c/10GHz=3cm.
=#
    
    N = length(d)
    
    ## Transverse wavevectors within layers.
    em = e.*m
    kt = sqrt.(w^2*em .-kp^2) # The MATLAB 'sqrt' gives kt with real(kt)>=0 (Right-Half-Plane)
    
    # Sign choice within layers (should not affect T matrix).
    ik = real.(kt).<0
    if imag(w)==0
        @. ik |= (abs(real(kt))<1e-10) & (imag(kt)<0)
    elseif real(w)==0
        @. ik |= (abs(real(kt))<1e-10) & (imag(kt)*imag(w)<0)
    end
    kt[ik] *= -1

    # Sign choice for bottom & top semi-infinite layers.
    if bc isa String
        if bc[1]=='o' && imag(kt[1])<0; kt[1] *= -1; end
        if bc[2]=='o' && imag(kt[N])<0; kt[N] *= -1; end
        if bc[1]=='l' && real(kt[1])<0; kt[1] *= -1; end # [likely redundant, see above]
        if bc[2]=='l' && real(kt[N])<0; kt[N] *= -1; end # [likely redundant, see above]
    end
    
    ## Impedances of layers. 
    if  pol=='e'
        wem = w*m
        X = kt./wem # = Y for TE
        dX_de = 1 ./(2m.*X)
    elseif pol=='m'
        wem = w*e
        X = kt./wem; # = Z for TM
        dX_de = ((kp/w)^2 .- em/2)./(e.^3 .*X)
    end
    
    D = 2π*d.*wem
    
    ## Phases of layers.
    U = 2π*kt.*d # = X.*D
    
    ## semi-infinite (d=Inf) and 2d-conductivity (d=-Inf) layers
    i_inf = d.==Inf
    d[i_inf] .= 0; D[i_inf] .= 0; U[i_inf] .= 0
    
    i_cond = d.==-Inf
    kt[i_cond] .= NaN .+ NaN*im; X[i_cond] .= e[i_cond]
    D[i_cond] .= -Inf; U[i_cond] .= 0
    

    ## Calculate T matrix.
    if N==1
        if bc isa String && (bc=="ol" || bc=="lo")
            warning("Open boundaries of different type for single layer!")
        end
        
        ## Propagating through the layer.
        T0, dT0 = Propagation_T(U)
        T = [1 0; 0 1] * T0
        dT_de = dT0 *(π*d*m/kt)
        dT_dd = dT0 *(2π*kt)
        
    else
        # Reflection and Transmission coefficients.
        rX = X[2:end]./X[1:end-1] 

        # Without this, there was a jump between +/-im for rX<0.
        irX = abs.(imag.(rX)).<100 .*eps.(abs.(rX))
        rX[irX] .= real.(rX[irX])
    
        ## Propagating through the layers.
        dT_de = zeros(ComplexF64, 2,2,N)
        dT_dd = zeros(ComplexF64, 2,2,N)

        T, dPj = Propagation_T(U[1])
        if derivs
            Ps = zeros(ComplexF64, 2,2,N); dP_de = similar(Ps); dP_dd = similar(Ps)
            Is = zeros(ComplexF64, 2,2,N-1); dI_de = similar(Is); dI_de1 = similar(Is)
            Ts = zeros(ComplexF64, 2,2,N-1)

            cde = π*d.*m./kt
            cdd = 2π*kt

            Ps[:,:,1] = T
            dP_de[:,:,1] = dPj *cde[1]
            dP_dd[:,:,1] = dPj *cdd[1]
        end

        jj = 2
        while jj<=N
            Pj, dPj = Propagation_T(U[jj])
            if abs(X[jj])>1e-8 || jj==N
                Ij, dIj = Interface_T(rX[jj-1])
    
                if derivs
                    Ps[:,:,jj] = Pj
                    dP_de[:,:,jj] = dPj * cde[jj]
                    dP_dd[:,:,jj] = dPj * cdd[jj]

                    Is[:,:,jj-1] = Ij
                    dI_de[:,:,jj-1] = dIj * (dX_de[jj]/X[jj-1])
                    dI_de1[:,:,jj-1] = -dIj * (dX_de[jj-1]*rX[jj-1]/X[jj-1])

                    Ts[:,:,jj-1] = T
                end

                T *= Ij*Pj

            else # Does not resolve X=0 for first and last layers.
                T *= Interface_T_zeroX(X[jj-1],X[jj],X[jj+1],im*2π*D[jj]) * Pj
                jj += 1
            end
            jj += 1
        end
        # Multiplication with the last Pj=PN will not do anything for an open system,
        # but will be useful if periodic or externally connected.

        ## Calculate derivatives
        if derivs
            dT_de[:,:,N] = Ts[:,:,N-1]*( dI_de[:,:,N-1]*Ps[:,:,N] + Is[:,:,N-1]*dP_de[:,:,N] )
            dT_dd[:,:,N] = Ts[:,:,N-1]*Is[:,:,N-1]*dP_dd[:,:,N]
            Tr = Ps[:,:,N]

            jj = N-1
            while jj >= 2
                dT_de[:,:,jj] = 
                begin Ts[:,:,jj-1]*
                    (dI_de[:,:,jj-1]*Ps[:,:,jj]*Is[:,:,jj] +
                        Is[:,:,jj-1]*dP_de[:,:,jj]*Is[:,:,jj] +
                        Is[:,:,jj-1]*Ps[:,:,jj]*dI_de1[:,:,jj]) * Tr
                end
                dT_dd[:,:,jj] = Ts[:,:,jj-1]*Is[:,:,jj-1]*dP_dd[:,:,jj]*Is[:,:,jj] * Tr
                Tr = Ps[:,:,jj]*Is[:,:,jj] * Tr
                jj -= 1
            end
            dT_de[:,:,1] = (dP_de[:,:,1]*Is[:,:,1] + Ps[:,:,1]*dI_de1[:,:,1]) * Tr
            dT_dd[:,:,1] = dP_dd[:,:,1]*Is[:,:,1] * Tr
            Tr = Ps[:,:,1]*Is[:,:,1] * Tr # just to check that T is the same

        end
    end

    return T, dT_de, dT_dd, X, kt
end

function Propagation_T(u)
    # Using the physics exp(-iwt) convention,
    # so remember that we must have imag(neff)>0 <=> imag(u)>0.
    expiu = exp(im*u); i_expiu = 1 ./expiu
    T  = [    i_expiu 0; 0    expiu]
    dT = [-im*i_expiu 0; 0 im*expiu]
    return T, dT
end    

function Interface_T(rX)
    # rX = X2/X1, where Xn = ktn/(w*mn) for 'e' and Xn = ktn/(w*en) for 'm'
    rX1 = 1 ./(1+rX)
    r = (1-rX).*rX1
    t = 2*sqrt(rX).*rX1; # t = sqrt(1-r^2);   # The minus sign would be needed here for a backward mode(?).
    it = 1 ./t; rt = r.*it
    T = [it rt; rt it]
    # Note that, for these definitions: T11*t = T22*t = t.^2+r.^2=1 always
    
    prX = rX-1; mrX = -rX-1
    dT = [prX mrX; mrX prX] / 4rX.^1.5

    return T, dT
end

function Interface_T_zeroX(x1, x2, x3, d2)
    iD = 0.5/sqrt(x1*x3)
    T11 = (x1+x3-d2*(x1*x3+x2^2))*iD
    T22 = (x1+x3+d2*(x1*x3+x2^2))*iD
    T12 = (x1-x3+d2*(x1*x3-x2^2))*iD
    T21 = (x1-x3-d2*(x1*x3-x2^2))*iD
    T = [T11 T12; T21 T22]
    return T
end
