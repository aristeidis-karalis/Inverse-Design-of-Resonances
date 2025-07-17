# -*- coding: utf-8 -*-
using Gmsh, GridapGmsh, Gridap, Gridap.Geometry, Gridap.Fields
using LinearAlgebra, KrylovKit
using LeastSquaresOptim
using JLD
using MPI
include("filter_params.jl")
include("QNMT.jl")

# Filter specifications
λo = 1     # Wavelength
N = 3     # filter order
fbandwidth = 0.01   # filter bandwidth in % of 1/λo
filtertype = "bandpass"   #["bandpass", "bandstop"]
profile = "ellip"    #["butter","cheby","icheby","ellip"]
pass = 0.25   # passband max attenuation in dB
stop = 25   # passband min attenuation in dB

fsr = 30*fbandwidth  # free spectral range

# Geometry, Materials & Mesh Parameters
n_hi = 3.4   # High-index material refractive index
n_lo = 1    # Low-index material refractive index
n_air = 1    # Air refractive index
μ = 1        # Magnetic permeability

H = 0.25       # Height of cell (half period) in wavelengths
D = 3.00       # Length of design region in wavelengths

Dair = 0.5     # Length of air region in wavelengths
Dpml = 0.5      # Length of PML in wavelengths
Rpml = 1e-12    # Tolerance for PML reflection

# Optimization hyperparameters
σ1phase = -1im    # phase of σ1

αC = 0.036          # multiplicative factors of C errors
sqrtM = 0           # extra multiplicative factor of C errors, choose 1, or 0 to divide by sqrt(M)

ptotLim_λ = []      # total-material constraint (use [] for no constraint)
γptot = 10          # multiplier for total-material constraint

λinit = 0.1       # Initial Levenberg-Marquardt coefficient

r = 0.02          # p-Filter radius
β, η = 8, 0.5     # p-Threshold steepness & center

# Optimization criteria
MAX_ITER = 30000        # maximum optimization iterations
X_TOL = 1e-12           # maximum parameters tolerance
F_TOL = 1e-12           # maximum objectives tolerance
G_TOL = 1e-12           # maximum objectives' derivative tolerance

# Saving resulting data
LSTSQ_p_TOL = 1e-4        # limit for change in sum squares of p, below which to save solution
LSTSQ_C_TOL = 5e-3        # limit for sum squares of C, below which to save solution

# Plotting frequencies
fpbw = 0.6         # bandwidth to plot final result
Nfplot = 3000      # frequency points to plot final result


### ------------------------------------------------------------------------------------
# MPI and job array initializations
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
mpisize = MPI.Comm_size(comm)

# Calculate some optimization parameters
M = mpisize - N
use_ptotLim = !isempty(ptotLim_λ)
ptotScale = N*H*(0.5/n_hi)
if use_ptotLim; ptotLim = ptotLim_λ * ptotScale; end

# Save optimization parameters
opt_params = (; r, β, η, αC, ptotLim_λ, γptot, fsr,M)


# Save physical parameters
H *= λo
D *= λo
Dair *= λo
L = D + 2*Dair
phys_params = (; n_hi, n_lo, n_air, μ, H, D, L, Dpml, Rpml)


## Create geometry and mesh
model = GmshDiscreteModel("TopOptRes.msh")   # Load mesh file

## FEM setup
order = 1
reffe = ReferenceFE(lagrangian, Float64, order)
V = TestFESpace(model, reffe,vector_type = Vector{ComplexF64})#dirichlet_tags = ["LR_Edge"],
U = V   # mathematically equivalent to TrialFESpace(V,0)

degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)
Ω_d = Triangulation(model, tags="Design")
dΩ_d = Measure(Ω_d, degree)
Γ_lr = BoundaryTriangulation(model; tags = ["LR_Edge"]) # Left Right boundary line
dΓ_lr = Measure(Γ_lr, degree)
Γ_dl = BoundaryTriangulation(model; tags = ["design_left"])
dΓ_dl = Measure(Γ_dl, degree)
Γ_dr = BoundaryTriangulation(model; tags = ["design_right"])
dΓ_dr = Measure(Γ_dr, degree)

p_reffe = ReferenceFE(lagrangian, Float64, 0)
Q = TestFESpace(Ω_d, p_reffe, vector_type = Vector{Float64})
P = Q
np = num_free_dofs(P)
pf_reffe = ReferenceFE(lagrangian, Float64, 1)
Qf = TestFESpace(Ω_d, pf_reffe, vector_type = Vector{Float64})
Pf = Qf

Pc = TestFESpace(Ω_d, p_reffe, vector_type = Vector{ComplexF64})
Pfc = TestFESpace(Ω_d, pf_reffe, vector_type = Vector{ComplexF64})

# Save mesh parameters
fem_params = (; V, U, Q, P, Qf, Pf, Pc, Pfc, np, Ω, dΩ, Ω_d, dΩ_d, dΓ_dl, dΓ_dr, dΓ_lr)


## Calculate poles and C of target spectral response
fcenter = 1/λo
(poles, t∞, r∞) = calc_poles(N, fcenter,fbandwidth, profile,filtertype, pass,stop)

# Choose a phase for the σs and adjust r∞
σs = [σ1phase * (-1)^i for i=0:N-1]
r∞ /= σ1phase   # t∞, r∞ are the target values of C21, C11

# Save filter specs and poles
filter_params = (; N, fcenter,fbandwidth, filtertype,profile, pass,stop, poles,σs,t∞,r∞)
if rank == 0; println("Optimizing topology for ",filter_params); end

# Initial geometry
if filtertype == "bandstop"
    pd0 = zeros(fem_params.np)
    
elseif filtertype == "bandpass"
    d1 = λo / 4n_hi
    d2 = λo / 4n_lo
    grating_periodicity = d1 + d2
    half_cavity_width = d1/2;

    function quarter_eps(x)
        if abs(x[1]) < half_cavity_width
            return 1
        else
            dx = abs(x[1]) - half_cavity_width
            while dx > grating_periodicity
                dx -= grating_periodicity
            end
            return (dx < d2) ? 0 : 1;
        end
    end

    disk(x) = quarter_eps(x)
    diskp = interpolate_everywhere(disk,P)
    pd0 = get_free_dof_values(diskp)
    pd0 = max.(0, min.(1, pd0))
end    

## PML setup
σPML = -3 / 4 * log(phys_params.Rpml) / phys_params.Dpml / phys_params.n_air / 2π
function s_PML(x; phys_params)
    u = abs.(Tuple(x)).-(phys_params.L, phys_params.H)./2
    return @. ifelse(u > 0,  1 + (1im * σPML) * (u / phys_params.Dpml)^2, $(1.0+0im))
end

function ds_PML(x; phys_params)
    u = abs.(Tuple(x)).-(phys_params.L, phys_params.H)./2
    ds = @. ifelse(u > 0, (2im * σPML) * (1 / phys_params.Dpml)^2 * u, $(0.0+0im))
    return ds.*sign.(Tuple(x))
end

struct Λ{PT} <: Function
    phys_params::PT
end

function (Λf::Λ)(x)
    s_x,s_y = s_PML(x; Λf.phys_params)
    return VectorValue(1/s_x, 1/s_y)
end

Fields.∇(Λf::Λ) = x -> TensorValue{2, 2, ComplexF64}(-(Λf(x)[1])^2 * ds_PML(x; Λf.phys_params)[1], 0, 0, -(Λf(x)[2])^2 * ds_PML(x; Λf.phys_params)[2])


## Filter, Threshold and Response functions
a_f(r, u, v) = r^2 * (∇(v) ⋅ ∇(u)) + (v * u)
Af = assemble_matrix(fem_params.Pf, fem_params.Qf) do u, v
    ∫(a_f(r, u, v))fem_params.dΩ_d
end
Af_chol = cholesky(Symmetric(Af)) # Af is real symmetric (Af = Af')

function pFilter(p0; r, fem_params)
    ph = FEFunction(fem_params.P, p0)
    p_vec = assemble_vector(v -> ∫(v * ph)fem_params.dΩ_d, fem_params.Pf)
    pf_vec = Af_chol \ p_vec
    return FEFunction(fem_params.Pf, pf_vec)
end

function pThreshold(pfh; β, η)
    return ((tanh(β * η) + tanh(β * (pfh - η))) / (tanh(β * η) + tanh(β * (1.0 - η))))
end


# p- and k-independent matrices
a_base(u, v; phys_params) = (1/phys_params.n_air^2) * ((∇ .* (Λ(phys_params) * v)) ⊙ (Λ(phys_params) .* ∇(u)))
A_Ω = assemble_matrix(fem_params.U, fem_params.V) do u, v
    ∫(a_base(u, v; phys_params))fem_params.dΩ
end
A_Ωk2 = assemble_matrix(fem_params.U, fem_params.V) do u, v
    ∫(phys_params.μ * (v * u))fem_params.dΩ
end
A_Γk = assemble_matrix(fem_params.U, fem_params.V) do u, v
    ∫(phys_params.μ * (v * u))fem_params.dΓ_lr
end

ξd(p, n_lo, n_hi) = 1 / (n_lo + (n_hi - n_lo) * p)^2 - 1 / phys_params.n_air^2 # in the design region
a_design(u, v, pth; phys_params) = ((p -> ξd(p, phys_params.n_lo, phys_params.n_hi)) ∘ pth) * (∇(v) ⊙ ∇(u))
function MatrixA(pth, A; phys_params, fem_params)
    A_Ωd = assemble_matrix(fem_params.U, fem_params.V) do u, v
        ∫(a_design(u, v, pth; phys_params))fem_params.dΩ_d
    end
    return lu(A_Ωd + A)
end


## Filter, Threshold and Response functions' derivatives
function DgDp_DgDpf(dgdpf; fem_params) #adjoint of filter
    w = Af_chol \ dgdpf   # Reminder: Af is real symmetric, dgdpf, w are col vectors
    sw = size(w); if length(sw)==1; col = 1; else; col = sw[2]; end
    elty = eltype(dgdpf)
    dgdp = zeros(elty, col,fem_params.np)
    if elty == Float64; Pf = fem_params.Pf; P = fem_params.P; elseif elty==ComplexF64; Pf = fem_params.Pfc; P = fem_params.Pc; end
    for i =1:col
        wh = FEFunction(Pfc, w[:,i])
        dgdp[i,:] = assemble_vector(dp -> ∫(wh * dp)fem_params.dΩ_d, Pc)
    end
    return dgdp   # is a Jacobian
end


DptDpf(pf, β, η) = β * (1.0 - tanh(β * (pf - η))^2) / (tanh(β * η) + tanh(β * (1.0 - η)))
DξDpf(pf, n_lo, n_hi, β, η)= 2 * (n_lo - n_hi) / (n_lo + (n_hi - n_lo) * pThreshold(pf; β, η))^3 * DptDpf(pf, β, η)

## Excitation and initial vectors
b_ls = assemble_vector(v->(∫(v)fem_params.dΓ_dl), fem_params.V)
b_rs = assemble_vector(v->(∫(v)fem_params.dΓ_dr), fem_params.V)
pzero = FEFunction(fem_params.Pf, zeros(fem_params.np))
pfh = FEFunction(fem_params.Pf, zeros(fem_params.np))
pth = FEFunction(fem_params.Pf, zeros(fem_params.np))

## Frequencies for C and contour integration
freqs = filter_params.fcenter*LinRange(1-fsr/2, 1+fsr/2, M)
if sqrtM != 1; sqrtM = sqrt(M); end

## Split tasks to nodes
if rank<N
    pfi = rank+1

    k = 2π*conj(filter_params.poles[pfi])   # excite system at the complex S-zero (conjugate of pole)
    A_ΩΓ = A_Ω - k^2 * A_Ωk2 - (1im*k) * A_Γk

    σ = conj(filter_params.σs[pfi])   # excitation must also be conjugate of the pole's
    b = b_ls + σ * b_rs

    A0_lu = MatrixA(pzero, A_ΩΓ; phys_params, fem_params)
    u0 = A0_lu \ [b_ls   σ * b_rs]
    s1 = dot(b_ls, u0[:,1])
    s2 = dot(b_rs, u0[:,2])

    A_lu = A0_lu
    u = similar(u0[:,1])

    function g_p(p0::Vector; r, β, η, phys_params, fem_params)
        global pfh = pFilter(p0; r, fem_params)
        global pth = (pf -> pThreshold(pf; β, η)) ∘ pfh
        
        global A_lu = MatrixA(pth, A_ΩΓ; phys_params, fem_params)
        global u = A_lu \ b
        
        ul = dot(b_ls, u) - s1
        ur = dot(b_rs, u) - s2
        g = [ul; ur]
        return [real(g); imag(g)]
    end
    
    function DgDp(p0::Vector; r, β, η, phys_params, fem_params)
        # The commented global variables below do not need to be calculated again, since DgDp is always calcuated after g_p
        # pfh = pFilter(p0; r, fem_params)
        # pth = (pf -> pThreshold(pf; β, η)) ∘ pfh
        # A_lu = MatrixA(pth, A_ΩΓ; phys_params, fem_params)
        # u = A_lu \ b

        dξdpf = (p -> DξDpf(p, phys_params.n_lo, phys_params.n_hi, β, η)) ∘ pfh

        vh = A_lu' \ [b_ls b_rs]
        vhconj_l = FEFunction(fem_params.U, conj(vh[:,1]))
        vhconj_r = FEFunction(fem_params.U, conj(vh[:,2]))
        
        Duh = ∇(FEFunction(fem_params.U, u))
        
        DADpf_l = -dξdpf * (∇(vhconj_l) ⊙ Duh)
        DADpf_r = -dξdpf * (∇(vhconj_r) ⊙ Duh)
        
        dgdpf_l = assemble_vector(dp -> ∫(DADpf_l * dp)fem_params.dΩ_d, fem_params.Pfc)
        dgdpf_r = assemble_vector(dp -> ∫(DADpf_r * dp)fem_params.dΩ_d, fem_params.Pfc)
        dgdpf = [dgdpf_l;; dgdpf_r]
        
        # Derivative of poles' objectives w.r.t. unfiltered parameters (adjoint of filter)
        dgdp = DgDp_DgDpf(dgdpf; fem_params)
        return [real(dgdp); imag(dgdp)]
    end
    
    if rank == 0
        paramsLen = fem_params.np + use_ptotLim
        p_history = zeros(Float64, MAX_ITER+1, paramsLen)
        outputLen = 4N + 2M + use_ptotLim
        f_history = zeros(Float64, MAX_ITER+1, outputLen)
        cur_iter = 0
        init_p_error = 0.0; init_C_error = 0.0
        ptot = 0.0

        function DptotDp(p0::Vector; r, β, η, fem_params)
            DpthDpf = (pf -> DptDpf(pf, β, η)) ∘ pfh
            dptotdpf = assemble_vector(v->(∫(v*DpthDpf)fem_params.dΩ_d), fem_params.Pf)
            return DgDp_DgDpf(dptotdpf; fem_params)
        end

        function lstsq_f!(out, x)
            global cur_iter += 1
            
            p_history[cur_iter,:] = x
            
            p0 = x[1:fem_params.np]
            for i = 1:mpisize-1
                MPI.Send([11], i, 5, comm)
                MPI.Send(p0, i, 0, comm)            
            end

            pvalue = g_p(p0; r, β, η, phys_params, fem_params)
            out[1:4] = pvalue

            for i = 2:N
                cvalue_4 = zeros(4)
                MPI.Recv!(cvalue_4, i-1, 11, comm)
                out[4i-3:4i] = cvalue_4
            end
            for i = 1:M
                cvalue = zeros(2)
                MPI.Recv!(cvalue, N+i-1, 11, comm)
                out[4N+2i-1:4N+2i] = cvalue * αC/sqrtM
            end
            
            global ptot = sum(∫(pth)fem_params.dΩ_d)
            if use_ptotLim; out[outputLen] = (ptot - x[paramsLen]) * γptot; end
            
            f_history[cur_iter,:] = out
            
            if mod(cur_iter,100)==1
                println("Current iteration = ", cur_iter)
                p_error = sum(out[1:4N].^2)
                C_error = sum(out[4N+1:4N+2M].^2)
                lstsq_error = p_error + C_error + use_ptotLim * out[outputLen]^2
                if cur_iter==1; global init_p_error = p_error; global init_C_error = C_error; end
                println("err = ",lstsq_error, ", with poles_err = ",p_error, ", C_err = ",C_error," and ptot = ",ptot/ptotScale)
            end
        end
        
        function lstsq_g!(J, x)
            p0 = x[1:fem_params.np]
            for i = 1:mpisize-1
                MPI.Send([12], i, 5, comm)
                MPI.Send(p0, i, 0, comm)
            end

            dgf = DgDp(p0; r, β, η, phys_params, fem_params)
            J[1:4,1:fem_params.np] = dgf

            for i = 2:N
                cvalue_4np = zeros(4*fem_params.np)
                MPI.Recv!(cvalue_4np, i-1, 12, comm)
                J[4i-3:4i,1:fem_params.np] = reshape(cvalue_4np,4,fem_params.np)
            end
            for i = 1:M
                cvalue_2np = zeros(2*fem_params.np)
                MPI.Recv!(cvalue_2np, N+i-1, 12, comm)
                value = reshape(cvalue_2np,2,fem_params.np)
                J[4N+2i-1:4N+2i, 1:fem_params.np] = value * αC/sqrtM
            end

            if use_ptotLim
                J[outputLen,1:fem_params.np] = DptotDp(p0; r, β, η, fem_params) * γptot
                J[outputLen,fem_params.np+1:paramsLen-1] .= 0
                J[outputLen,paramsLen] = -1 * γptot
                J[1:outputLen-1,paramsLen] .= 0
            end
        end

        ### PERFORM OPTIMIZATION
        # Specify initial parameter guess
        p_init = copy(pd0)
        if use_ptotLim
            pfh0 = pFilter(pd0; r, fem_params)
            pth0 = (pf -> pThreshold(pf; β, η)) ∘ pfh0
            ptot0 = sum(∫(pth0)fem_params.dΩ_d)
            println("ptot0 = ", ptot0)            
            push!(p_init, min(ptot0,ptotLim))
        end

        # Specify the parameter bounds
        lb = -Inf*ones(Float64, fem_params.np)
        ub =  Inf*ones(Float64, fem_params.np)
        if use_ptotLim; push!(lb,0); push!(ub,ptotLim); end

        # Run optimization
        lstsq = LeastSquaresProblem(x = p_init, f! = lstsq_f!, g! = lstsq_g!, output_length = outputLen)
        optimizer_solver = LeastSquaresOptim.LevenbergMarquardt(LeastSquaresOptim.Cholesky(true))   # with "false" you get original non-transposed solver
        timeElapsed = @elapsed LeastSquaresOptim.optimize!(lstsq, optimizer_solver, lower=lb, upper=ub, Δ=1/λinit, iterations=MAX_ITER, x_tol=X_TOL, f_tol=F_TOL, g_tol=G_TOL)

        # Notifications upon completion
        println("end of optimization")
        println("total optimization time = ", timeElapsed)
        println("final iteration = ", cur_iter)
        println("optimization time per iteration = ", timeElapsed/cur_iter)
        
        final_p_error = sum(lstsq.y[1:4N].^2)
        final_C_error = sum(lstsq.y[4N+1:4N+2M].^2)
        final_error = final_p_error + final_C_error + use_ptotLim * lstsq.y[outputLen]^2
        println("final error = ",final_error, ", with final_poles_error = ", final_p_error, ", final_C_error = ", final_C_error, " and final_ptot = ", ptot)

        p_history = p_history[1:cur_iter,:]
        f_history = f_history[1:cur_iter,:]
        final_p = p_history[end,1:fem_params.np]

        @show (σ1phase, αC)

        # Tell other nodes to stop and decide if to save results
        f_or_df = (final_p_error/init_p_error < LSTSQ_p_TOL) && (final_C_error < LSTSQ_C_TOL) ? [100] : [200]

        for i=1:mpisize-1
            MPI.Send(f_or_df, i, 5, comm)
        end        
        
        
        
        ## Calculate poles of final result
        final_pfh = pFilter(final_p; r, fem_params)
        final_pth = (pf -> pThreshold(pf; β, η)) ∘ final_pfh

        k = 2π*filter_params.fcenter
        A_ΩΓ = A_Ω - k^2 * A_Ωk2 - (1im*k) * A_Γk
        A = MatrixA(final_pth, A_ΩΓ; phys_params, fem_params)

        vals, final_vecs, info = eigsolve(x-> A\(A_Ωk2*x), rand(ComplexF64, size(A, 1)),  3N, tol = 0);
        final_poles = sqrt.(1 ./vals .+ k^2)/2π

        final_σs = similar(final_poles)
        for i = 1:length(final_poles)
            final_σs[i] = dot(b_rs, final_vecs[i]) / dot(b_ls, final_vecs[i])
        end

        println("filter poles = ",filter_params.poles)
        sorted_poles = final_poles[sortperm(abs.(final_poles .- filter_params.fcenter))[1:N]]
        show_poles = sorted_poles[sortperm(real(sorted_poles))]
        println("final_poles = ",show_poles)

        
    else
        # These nodes enforce the poles 2:N
        while true
            global f_or_df = [0]
            MPI.Recv!(f_or_df, 0, 5, comm)
            if f_or_df[1] == 11
                cur_p0 = zeros(fem_params.np)
                MPI.Recv!(cur_p0, 0, 0, comm)
                
                cvalue_4 = g_p(cur_p0; r, β, η, phys_params, fem_params)
                MPI.Send(cvalue_4, 0, 11, comm)

            elseif f_or_df[1] == 12
                cur_p0 = zeros(fem_params.np)
                MPI.Recv!(cur_p0, 0, 0, comm)

                cvalue_4np = DgDp(cur_p0; r, β, η, phys_params, fem_params)
                MPI.Send(reshape(cvalue_4np,1,4*fem_params.np), 0, 12, comm)

            elseif f_or_df[1] >= 100
                break
            else
                println("Err 583", f_or_df)
                break
            end
        end
    end    
else
    # These nodes enforce the C constraint at frequency fi
    fi = freqs[rank-N+1]

    # Calculate inv(Sbar(poles)) at C-constraint frequencies
    Sbar, = calc_Sbar(fi, filter_params.poles,  filter_params.σs, 2)
    iSbar = inv(Sbar)

    k = 2π*fi
    A_ΩΓ = A_Ω - k^2 * A_Ωk2 - (1im*k) * A_Γk
    dAdk = 2k * A_Ωk2 + 1im * A_Γk   # this is -dAdk (see later)
    
    A0_lu = MatrixA(pzero, A_ΩΓ; phys_params, fem_params)
    u0 = A0_lu \ [b_ls b_rs]
    sp1 = dot(b_ls, u0[:,1])
    sp2 = dot(b_rs, u0[:,2])
    bl1 = b_ls/sp1
    br2 = b_rs/sp2

    AC_lu = A0_lu
    uC = similar(u0)
    S = zeros(ComplexF64, 2,2)
    iS = zeros(ComplexF64, 2,2)
    C = zeros(ComplexF64, 2,2)
    r∞ = filter_params.r∞; t∞ = filter_params.t∞
    
    function S_pth(pth)
        global AC_lu = MatrixA(pth, A_ΩΓ; phys_params, fem_params)
        global uC = AC_lu \ [bl1 br2]   # this makes uC normalized by sp1 and sp2
        
        S11 = dot(b_ls, uC[:,1]) - 1
        S21 = dot(b_rs, uC[:,1])
        S12 = dot(b_ls, uC[:,2])
        S22 = dot(b_rs, uC[:,2]) - 1
        return [S11 S12; S21 S22]
    end

    function cc_p(p0::Vector; r, β, η, phys_params, fem_params)
        global pfh = pFilter(p0; r, fem_params)
        global pth = (pf -> pThreshold(pf; β, η)) ∘ pfh
        
        global S = S_pth(pth)
        global C = iSbar * S
        
        # reminder: r∞ is complex, t∞ is real (but works for overall C∞ phase too)
        cc = t∞ < 0.5 ? C[2,1]/conj(r∞) - t∞/conj(C[1,1]) : conj(C[1,1])/t∞ - conj(r∞)/C[2,1]
        return [real(cc); imag(cc)]
    end

    function DccDp(p0::Vector; β, η, phys_params, fem_params)
        # The commented global variables below do not need to be calculated again, since DccDp is always calcuated after cc_p
        #pfh = pFilter(p0; r, fem_params)
        #pth = (pf -> pThreshold(pf; β, η)) ∘ pfh
        #AC_lu = MatrixA(pth, A_ΩΓ; phys_params, fem_params)
        #uC = AC_lu \ b_ls

        dξdpf = (p -> DξDpf(p, phys_params.n_lo, phys_params.n_hi, β, η)) ∘ pfh

        vh =  AC_lu' \ [b_ls b_rs]
        vhconj1 = FEFunction(fem_params.U, conj(vh[:,1]))
        vhconj2 = FEFunction(fem_params.U, conj(vh[:,2]))

        Duh1 = ∇(FEFunction(fem_params.U, uC[:,1]))

        DADpf11 = -dξdpf * (∇(vhconj1) ⊙ Duh1)
        DADpf21 = -dξdpf * (∇(vhconj2) ⊙ Duh1)

        ## Calculate derivatives of S
        dS11 = assemble_vector(dp -> ∫(DADpf11 * dp)fem_params.dΩ_d, fem_params.Pfc)
        dS21 = assemble_vector(dp -> ∫(DADpf21 * dp)fem_params.dΩ_d, fem_params.Pfc)
        
        ## Calculate derivatives of C
        dC11 = iSbar[1,1] * dS11 + iSbar[1,2] * dS21
        dC21 = iSbar[2,1] * dS11 + iSbar[2,2] * dS21
        
        # Calculate derivatives of cc objective
        dccdpf = dC21/conj(r∞) + t∞*conj(dC11/C[1,1]^2)

        # Calculate derivatives of cc w.r.t. unfiltered parameters
        dccdp = DgDp_DgDpf(dccdpf; fem_params)
        return [real(dccdp); imag(dccdp)]
    end

    while true    
        global f_or_df = [0]
        MPI.Recv!(f_or_df, 0, 5, comm)
        if f_or_df[1] == 11
            cur_p0 = zeros(fem_params.np)
            MPI.Recv!(cur_p0, 0, 0, comm)

            cvalue = cc_p(cur_p0; r, β, η, phys_params, fem_params)
            MPI.Send(cvalue, 0, 11, comm)

        elseif f_or_df[1] == 12
            cur_p0 = zeros(fem_params.np)
            MPI.Recv!(cur_p0, 0, 0, comm)
            
            cvalue_2np = DccDp(cur_p0; β, η, phys_params, fem_params)
            MPI.Send(reshape(cvalue_2np,1,2*fem_params.np), 0, 12, comm)

        elseif f_or_df[1] >= 100
            break
        else
            println("Err 583", f_or_df)
            break
        end
    end
end


### ------------------------------------------------------------------------------
### If optimization successful, save data and calculate response
if f_or_df[1] == 100

    if rank == 0
        # Create filename
        bstr = string(Int(round((1+filter_params.fbandwidth)*100)))[2:3]
        fstr = "_bw"*bstr*"N"*string(N)
        if M>0
            fsrstr = string(Int(round((1+fsr)*100)))[2:3]
            fstr *= "_fsr"*fsrstr*"M"*string(M)*"Cf"*string(opt_params.αC)[3:end]
            if sqrtM != 1; fstr *= "sqrtM"; end
        end
        if use_ptotLim; fstr *= "_pt"*string(ptotLim_λ) * "gp"*string(γptot); end
        
        # Save optimization data
        save("TopOpt"*fstr*".jld", "filter_params", filter_params, "phys_params", phys_params, "fem_params", fem_params, "opt_params", opt_params, "final_p", final_p, "final_ptot", ptot/ptotScale, "final_error", final_error, "timeElapsed", timeElapsed)
        save("TopOpt"*fstr*"_h.jld", "p_history", p_history, "f_history", f_history)

        # Send final p to all nodes
        for i=1:mpisize-1
            MPI.Send(final_p, i, 7, comm)
        end
    else
        final_p = zeros(fem_params.np)
        MPI.Recv!(final_p, 0, 7, comm)

        final_pfh = pFilter(final_p; r, fem_params)
        final_pth = (pf -> pThreshold(pf; β, η)) ∘ final_pfh
    end

    # Calculate transmission of final result
    fpl = filter_params.fcenter*(1 - fpbw/2)
    fpr = filter_params.fcenter*(1 + fpbw/2)
    Nfplot_perNode = div(Nfplot,mpisize)
    Nfplot = Nfplot_perNode*mpisize
    plot_freqs = LinRange(fpl, fpr, Nfplot)

    function Sx1(q, pth; β, η, phys_params, fem_params)
        A_ΩΓ = A_Ω - q^2 * A_Ωk2 - (1im*q) * A_Γk

        A0_lu = MatrixA(pzero, A_ΩΓ; phys_params, fem_params)
        u0 = A0_lu \ b_ls
        sp1 = dot(b_ls, u0)

        A = MatrixA(pth, A_ΩΓ; phys_params, fem_params)
        u = A \ b_ls

        S11 = dot(b_ls, u)/sp1 - 1
        S21 = dot(b_rs, u)/sp1

        return S11, S21
    end

    S1temp = zeros(ComplexF64, Nfplot_perNode)
    S2temp = zeros(ComplexF64, Nfplot_perNode)
    C1temp = zeros(ComplexF64, Nfplot_perNode)
    C2temp = zeros(ComplexF64, Nfplot_perNode)
    P1temp = zeros(ComplexF64, Nfplot_perNode)
    P2temp = zeros(ComplexF64, Nfplot_perNode)
    for i = 1:Nfplot_perNode
        Sfi = Nfplot_perNode*rank + i
        local k = 2π*plot_freqs[Sfi]

        S1temp[i], S2temp[i] = Sx1(k, final_pth; β, η, phys_params, fem_params)

        Sbar_i, = calc_Sbar(plot_freqs[Sfi], filter_params.poles,  filter_params.σs, 2)

        C_i = Sbar_i \ [S1temp[i]; S2temp[i]]
        C1temp[i] = C_i[1]; C2temp[i] = C_i[2]

        P_i = Sbar_i * [filter_params.r∞; filter_params.t∞]   # First column of ideal C∞
        P1temp[i] = P_i[1]; P2temp[i] = P_i[2]
    end

    if rank == 0
        S11 = zeros(ComplexF64, Nfplot)
        S21 = zeros(ComplexF64, Nfplot)
        C11 = zeros(ComplexF64, Nfplot)
        C21 = zeros(ComplexF64, Nfplot)
        P11 = zeros(ComplexF64, Nfplot)
        P21 = zeros(ComplexF64, Nfplot)

        S11[1:Nfplot_perNode] = S1temp
        S21[1:Nfplot_perNode] = S2temp
        C11[1:Nfplot_perNode] = C1temp
        C21[1:Nfplot_perNode] = C2temp
        P11[1:Nfplot_perNode] = P1temp
        P21[1:Nfplot_perNode] = P2temp
        SCtemp = zeros(ComplexF64, 6*Nfplot_perNode)
        for i=1:mpisize-1
            MPI.Recv!(SCtemp, i, 8, comm)
            S11[i*Nfplot_perNode .+ (1:Nfplot_perNode)] = SCtemp[                 1 :   Nfplot_perNode]
            S21[i*Nfplot_perNode .+ (1:Nfplot_perNode)] = SCtemp[  Nfplot_perNode+1 : 2*Nfplot_perNode]
            C11[i*Nfplot_perNode .+ (1:Nfplot_perNode)] = SCtemp[2*Nfplot_perNode+1 : 3*Nfplot_perNode]
            C21[i*Nfplot_perNode .+ (1:Nfplot_perNode)] = SCtemp[3*Nfplot_perNode+1 : 4*Nfplot_perNode]
            P11[i*Nfplot_perNode .+ (1:Nfplot_perNode)] = SCtemp[4*Nfplot_perNode+1 : 5*Nfplot_perNode]
            P21[i*Nfplot_perNode .+ (1:Nfplot_perNode)] = SCtemp[5*Nfplot_perNode+1 : 6*Nfplot_perNode]
        end

        println("end of calculations")
        save("TopOpt"*fstr*"_response.jld", "final_poles", final_poles, "final_σs", final_σs, "final_vecs", final_vecs, "plot_fs", plot_freqs, "S11", S11, "S21", S21, "C11", C11, "C21", C21, "P11", P11, "P21", P21)

    else
        MPI.Send([S1temp; S2temp; C1temp; C2temp; P1temp; P2temp], 0, 8, comm)
    end
else
    if rank == 0; println("Not saving result. End of code."); end
end
