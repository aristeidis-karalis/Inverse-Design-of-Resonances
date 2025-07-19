using Plots
using LinearAlgebra
using LeastSquaresOptim
include("filter_params.jl")
include("QNMT.jl")
include("LCladder.jl")
include("ABCD.jl")

function LCladder_design(N,bw,profile,pass,stop,σ1phase, RG,RL,LC0; ω_plot=0.95:0.00025:1.05)

    # Target filter parameters
    (poles, t∞, r∞) = calc_poles(N, 1,bw, profile,"bandpass", pass,stop)
    res = [poles (-1).^[0:N-1;] * exp(im*σ1phase)]
    println("Target filter poles and σs = ", res)

    ## LC vectors
    Np = div(length(LC0),2)
    println("Number of elements = ", 2Np)

    lc0 = deepcopy(LC0)
    lc0[2:2:Np] .= 1 ./ LC0[2:2:Np]
    lc0[Np+1:2:2Np] .= 1 ./ LC0[Np+1:2:2Np]
    l0 = lc0[1:Np]
    c0 = lc0[Np+1:2Np]

    # Parameter bounds
    lb = zeros(2Np)
    ub = 1e5*ones(2Np)

    # Optimization functions
    global cur_iter = 0

    function lstsq_f!(F, x)
        global cur_iter += 1
        Fi, _ = allConstraints(x,res,RG,RL)
        F[:] .= Fi[:]
        if mod(cur_iter,100)==1
            println("Iteration = ", cur_iter, ", SumSq = ",sum(abs2, Fi))
        end
    end
    outputLen = 4N

    function lstsq_g!(J, x)
        _, Ji = allConstraints(x,res,RG,RL)
        J[:] .= Ji[:]
    end
    paramsLen = 2Np

    # Perform optimization
    lstsq = LeastSquaresProblem(x = lc0, f! = lstsq_f!, g! = lstsq_g!, output_length = outputLen)
    optimizer_solver = LeastSquaresOptim.LevenbergMarquardt(LeastSquaresOptim.Cholesky(true))
    LeastSquaresOptim.optimize!(lstsq, optimizer_solver; x_tol=1e-16, f_tol=1e-16, g_tol=1e-30, Δ=10, iterations=1000, lower=lb, upper=ub, show_trace=false)

    lcOpt = deepcopy(lstsq.x)[1:2Np]
    lOpt = lcOpt[1:Np]
    cOpt = lcOpt[Np+1:2Np]
    FOpt = deepcopy(lstsq.y)
    println("Iteration = ", cur_iter, ", SumSq = ",sum(abs2, FOpt))
    

    ## Plot initial guess and optimal solution
    Nω_plot = length(ω_plot)

    Sb = zeros(ComplexF64, 2,2,Nω_plot)
    for i=1:Nω_plot; Sb[:,:,i], _ = calc_Sbar(ω_plot[i],res[:,1],res[:,2],2); end
    # Sb = -Sb # In QNMT, should be C = I

    S0 = LCladder_S(ω_plot,l0,c0,RG,RL)
    plt0Sm, plt0Sa = plot_S(ω_plot,S0[2,1,:],Sb[2,1,:])

    SOpt = LCladder_S(ω_plot,lOpt,cOpt,RG,RL)
    pltOptSm, pltOptSa = plot_S(ω_plot,SOpt[2,1,:],Sb[2,1,:])

    LCOpt = deepcopy(lcOpt)
    LCOpt[2:2:Np] .= 1 ./ LCOpt[2:2:Np]
    LCOpt[Np+1:2:2Np] .= 1 ./ LCOpt[Np+1:2:2Np]

    return LCOpt, SOpt, Sb, pltOptSm, plt0Sm, pltOptSa, plt0Sa
end

function plot_S(ω,S21,Sb21)
    S21_dB = 20*log10.(abs.(S21))
    plt_Sm = Plots.plot(ω,S21_dB, label="S")
    if !isempty(Sb21)
        Sb21_dB = 20*log10.(abs.(Sb21))
        Plots.plot!(plt_Sm, ω,Sb21_dB, label="target")
    end
    Plots.xlims!(ω[1], ω[end]); #Plots.ylims!(-70, 0)
    Plots.xlabel!("frequency"); Plots.ylabel!("S_{21} [dB]")

    plt_Sa = Plots.plot(ω,unwrap(angle.(S21)/π; range=2).-2, label="S")
    if !isempty(Sb21)
        Plots.plot!(plt_Sa, ω,unwrap(angle.(Sb21)/π; range=2).-2, label="target")
    end
    Plots.xlims!(ω[1], ω[end]); #Plots.ylims!(-2, 2)
    Plots.xlabel!("frequency"); Plots.ylabel!("S_{21} phase")

    return plt_Sm, plt_Sa
end

###########################################################################
function allConstraints(x,res,RG,RL)
    Np = div(length(x),2)
    N = size(res,1)
    pole_fac = 1

    wo = conj(res[:,1]); σ21o = conj(res[:,2])

    S = zeros(ComplexF64, 2,2,N)
    dS = zeros(ComplexF64, 2,2,N,2Np)
    out_left = zeros(ComplexF64, N)
    out_right = zeros(ComplexF64, N)
    dout_left = zeros(ComplexF64, N,2Np)
    dout_right = zeros(ComplexF64, N,2Np)
    for i=1:N
        Mi, dMi_dl, dMi_dc = LCladder_M_dM(-im*wo[i],x[1:Np],x[Np+1:2Np]; derivs=true)
        Si, dSi = M2S(Mi,[dMi_dl;;; dMi_dc],RG,RL)
        S[:,:,i] .= Si
        dS[:,:,i,:] .= dSi

        out_left[i] = S[1,1,i] + σ21o[i] * S[1,2,i]
        out_right[i] = S[2,1,i] + σ21o[i] * S[2,2,i]
        dout_left[i,:] = dS[1,1,i,:] + σ21o[i] * dS[1,2,i,:]
        dout_right[i,:] = dS[2,1,i,:] + σ21o[i] * dS[2,2,i,:]
    end
    out = [out_left; out_right]
    F = [real(out); imag(out)]*pole_fac
    dout = [dout_left; dout_right]
    J = [real(dout); imag(dout)]*pole_fac

    return F, J
end

LC, S, Sb, pltSm, _, pltSa, _ = LCladder_design(4,0.01,"cheby",0.25,25,0, 1,1.6196,[100; 0.01; 100; 0.01; 100; 0.01; 100; 0.01; 100; 0.01]; ω_plot=0.95:0.00025:1.05);
pltSa
# standard phase is σ1phase=-π/2