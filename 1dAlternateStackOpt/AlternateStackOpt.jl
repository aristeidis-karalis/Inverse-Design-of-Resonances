using Plots
using LinearAlgebra
using LeastSquaresOptim
push!(LOAD_PATH,"../")
include("filter_params.jl")
include("QNMT.jl")
include("STmatrices.jl")
include("layered1D_T.jl")

function AlternateStack_d_design(N,bw,profile,pass,stop,σ1phase, n,pol,d0,es, Lnlim,γptot,ωc,αC,dlims; ω_plot = 0.5:0.00025:1.5)

    # Target filter parameters
    (poles, t∞, r∞) = calc_poles(N, 1,bw, profile,"bandpass", pass,stop)
    res = [poles (-1).^[0:N-1;] * exp(im*σ1phase)]
    println("Target filter poles and σs = ", res)

    dmin = dlims[1]; dmax = dlims[2]
    e = n.^2
    M = length(ωc)

    ## Layer Thicknesses and Materials vectors
    if length(d0) == 1
        P = d0
        L1 = 0.25/n[1]
        L2 = 0.25/n[2]
        d0 = [repeat([L1; L2],P); L1]

        es = [1; repeat([e[1]; e[2]],P); e[1]; e[2]] # Last entry is e[2] for asymmetric substrate
    end
    Nd = length(d0)
    println("Number of layers = ", Nd)

    ms = ones(Nd+2)

    i1 = findall(es[2:end-1].==e[1]); i2 = findall(es[2:end-1].==e[2])

    ## Optimization objectives and contraints

    # Parameter bounds
    lb = dmin*ones(Nd)
    ub = dmax*ones(Nd)
    ub[i1] /= n[1]; ub[i2] /= n[2]     # Scale upper bounds by refractive index

    d_scale = N/(sum(real(res[:,1]))/N) # wavelength, times number of modes
    if !isempty(Lnlim)
        Lnlim *= d_scale/n[1]
        d0 = [d0; min(sum(d0[i1]),Lnlim)]
        lb = [lb; 0]
        ub = [ub; Lnlim]
    end

    # Optimization functions
    global cur_iter = 0

    function lstsq_f!(F, x)
        global cur_iter += 1
        Fi, _ = allConstraints(x,es,ms,res,pol,ωc,αC,Lnlim,γptot)
        F[:] .= Fi[:]
        if mod(cur_iter,10000)==1
            println("Iteration = ", cur_iter, ", SumSq = ",sum(abs2, Fi)," & L = ",sum(x[1:Nd])/d_scale*n[2]," & Ln = ",sum(x[i1])/d_scale*n[1])
        end
    end
    outputLen = 4N + 2M + !isempty(Lnlim)

    function lstsq_g!(J, x)
        _, Ji = allConstraints(x,es,ms,res,pol,ωc,αC,Lnlim,γptot)
        J[:] .= Ji[:]
    end
    paramsLen = Nd + !isempty(Lnlim)

    # Perform optimization
    lstsq = LeastSquaresProblem(x = d0, f! = lstsq_f!, g! = lstsq_g!, output_length = outputLen)
    optimizer_solver = LeastSquaresOptim.LevenbergMarquardt(LeastSquaresOptim.Cholesky(true))
    LeastSquaresOptim.optimize!(lstsq, optimizer_solver; Δ=10, x_tol=1e-16, f_tol=1e-16, g_tol=1e-30, iterations=10000, lower=lb, upper=ub, show_trace=false)
    dsOpt = deepcopy(lstsq.x)[1:Nd]
    FOpt = deepcopy(lstsq.y)

    println("Iteration = ", cur_iter, ", SumSq = ",sum(abs2, FOpt)," & LOpt = ",sum(dsOpt[1:Nd])/d_scale*n[2], " and LnOpt = ",sum(dsOpt[i1])/d_scale*n[1])
    

    ## Plot initial guess and optimal solution
    Nω_plot = length(ω_plot)

    Sb = zeros(ComplexF64, 2,2,Nω_plot)
    T0 = zeros(ComplexF64, 2,2,Nω_plot)
    S = zeros(ComplexF64, 2,2,Nω_plot)
    C = zeros(ComplexF64, 2,2,Nω_plot)
    for i=1:Nω_plot
        Sb[:,:,i], _ = calc_Sbar(ω_plot[i],res[:,1],res[:,2],2)

        T0[:,:,i], _ = layered_iso_T_dT(ω_plot[i],0,es,ms,[Inf; d0[1:Nd]; Inf],"ll",pol)

        Ti, _ = layered_iso_T_dT(ω_plot[i],0,es,ms,[Inf; dsOpt[1:Nd]; Inf],"ll",pol)
        Si = T2S(Ti)
        S[:,:,i] = Si
        C[:,:,i] = Sb[:,:,i] \ Si
    end
    plt0d, plt0S = plot_dS(d0,es,ω_plot,Sb[2,1,:],1 ./T0[1,1,:],[])
    pltOptd, pltOptS = plot_dS(dsOpt,es,ω_plot,Sb[2,1,:],S[2,1,:],C[2,1,:])

    return dsOpt, S,C,Sb, es, res, FOpt, pltOptd, pltOptS, plt0d, plt0S
end

function d_clean(ds,es,dcutoff)
# Removes layers with thickness less than dcutoff
# It also combines the adjacent layers, if they have the same material

    d1 = deepcopy(ds); e1 = deepcopy(es)
    i = 1
    while i<length(d1) # length(d1) is changing through the while loop!
        if d1[i]<dcutoff
            deleteat!(d1,i); deleteat!(e1,i+1)
            if e1[i+1] == e1[i]
                if i>1 && d1[i]>=dcutoff; d1[i-1] += d1[i]; end
                deleteat!(d1,i); deleteat!(e1,i+1)
            end
        else
            i += 1
        end
    end
    Nd = length(d1)
    if d1[Nd]<dcutoff; deleteat!(d1,Nd-1:Nd); deleteat!(e1,Nd:Nd+1); end
    return d1, e1
end

function plot_dS(d,e,ω,Sb21,S21,C21)
    # Plot structure
    Nd = length(d)
    dinf = 0.5
    d_plot = [-dinf; 0; 0; [cumsum(d)[ceil(Int, i/2)] for i=1:2Nd]; sum(d)+dinf]
    e_plot = [transpose(e); transpose(e)][:]
    y = sqrt.(e_plot)

    plt_d = Plots.plot(d_plot,y, legend = false, linewidth=2.5,color=:orange,size=(800,300), font=:Times, bottommargin=5Plots.mm,leftmargin=5Plots.mm)
    Plots.xlims!(-dinf,sum(d)+dinf); Plots.ylims!(0.9, maximum(y)+0.1)
    Plots.xlabel!("x/λ"); Plots.ylabel!("Index of refraction")

    # Plot transmission
    Sb21_dB = 20*log10.(abs.(Sb21))
    plt_S = Plots.plot(ω,Sb21_dB, linewidth=10,color=:black,legend = false, ylim=(-81, 1), yticks = [0, -20, -40, -60, -80], xtickfontsize=15, ytickfontsize=15, size=(800,550), font=:Times, bottommargin=5Plots.mm,leftmargin=5Plots.mm, xlabel="Frequency f", ylabel = "\$|S_{21}|^2\$ (dB)",xlabelfontsize=15,ylabelfontsize=15)
    #Plots.xlabel!("Frequency f"); Plots.ylabel!("\$|S_{21}|^2\$ (dB)")

    S21_dB = 20*log10.(abs.(S21))
    Plots.plot!(plt_S,ω,S21_dB, linewidth=2.5,color=:darkorange)

    if !isempty(C21)
        C21_dB = 20*log10.(abs.(C21))
        Plots.plot!(plt_S,ω,C21_dB, linewidth=2.5,linestyle=:dash,color=:blue)
    end
   
    return plt_d, plt_S
end

###########################################################################
function allConstraints(x,es,ms,res,pol,ωc,αC,Lnlim,γptot)
    Np = length(x)
    Nd = length(es)-2
    N = size(res,1)

    d = x[1:Nd]
    nlim = (Np==Nd+1)
    wo = conj(res[:,1]); σ21o = conj(res[:,2])

    T = zeros(ComplexF64, 2,2,N)
    dT = zeros(ComplexF64, 2,2,N,Nd)
    for i=1:N
        Ti, _, dTi = layered_iso_T_dT(wo[i],0,es,ms,[Inf; d; Inf],"ll",pol; derivs=true)
        T[:,:,i] .= Ti
        dT[:,:,i,:] .= dTi[:,:,2:Nd+1]
    end
    T11 = T[1,1,:]
    T12 = T[1,2,:]
    T21 = T[2,1,:]
    T22 = T[2,2,:]
    out_left = (T21+σ21o)./T11
    out_right = (1 .-σ21o.*T12)./T11
    out = [out_left; out_right]
    F = [real(out); imag(out)]
    
    dout_left = zeros(ComplexF64, N,Nd)
    dout_right = zeros(ComplexF64, N,Nd)
    for i=1:N
        dT11 = dT[1,1,i,:]
        dT12 = dT[1,2,i,:]
        dT21 = dT[2,1,i,:]
        dT22 = dT[2,2,i,:]
        dout_left[i,:] = (dT21 - out_left[i]*dT11)/T11[i]
        dout_right[i,:] = -(σ21o[i]*dT12 + out_right[i]*dT11)/T11[i]
    end
    dout = [dout_left; dout_right]
    dout = [real(dout); imag(dout)]
    J = [dout;;   zeros(4N,Np-Nd)]   # pole objectives do not depend on slack variables

    # limit for length of n region
    if nlim
        i1 = findall(es[2:end-1].==maximum(es))   # layers with high index
        Ln = sum(d[i1]) - x[Nd+1]

        dLn = zeros(1,Nd+1)
        dLn[1,i1] .= 1    # derivative at high index
        dLn[1,Nd+1] = -1     # derivative w.r.t. slack variable

        F = [F; Ln*γptot]
        J = [J; dLn*γptot]
    end

    if !isempty(ωc)
        M = length(ωc)
        C21 = zeros(ComplexF64, M)
        dC21_dd = zeros(ComplexF64, M,Nd)
        for i=1:M
            Tc, ~, dTc = layered_iso_T_dT(ωc[i],0,es,ms,[Inf; d[1:Nd]; Inf],"ll",pol; derivs=true)
            Sb, _ = calc_Sbar(ωc[i],res[:,1],res[:,2],2)
    
            Sc21 = 1/Tc[1,1]
            Sc11 = Tc[2,1]*Sc21
            iSb = inv(Sb)
            C21[i] = iSb[2,1]*Sc11 + iSb[2,2]*Sc21

            dT11 = dTc[1,1,2:Nd+1]
            dT21 = dTc[2,1,2:Nd+1]
            dSc11 = (dT21 - Sc11*dT11)*Sc21
            dSc21 = - Sc21^2*dT11
            dC21_dd[i,:] = iSb[2,1]*dSc11 + iSb[2,2]*dSc21
        end
        Cout = [real(C21); imag(C21)]
        dCout = [[real(dC21_dd); imag(dC21_dd)];; zeros(2Nc,Np-Nd)]

        F = [F; Cout*αC/sqrt(M)]
        J = [J; dCout*αC/sqrt(M)]
    end

    return F, J
end

d1,S1,C1,Sb1,e1,res,F1,pltd1,pltS1,pltd0,pltS0 = AlternateStack_d_design(3,0.01,"cheby",0.25,25,0, [3.4 1.4],'m',14,[], 1.5,10,[],[],[0;0.75]);
d1c,e1c = d_clean(d1,e1,0.02);
d2,S2,C2,Sb2,e2,res,F2,pltd2,pltS2,_,_ = AlternateStack_d_design(3,0.01,"cheby",0.25,25,0, [3.4 1.4],'m',d1c,e1c, 1.5,10,[],[],[0;0.75]);
pltS2
