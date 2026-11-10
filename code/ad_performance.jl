# Runtime and error of the Jacobian ∂P(kᵢ)/∂θⱼ of the matter power spectrum with respect to N parameters,
# computed with one-sided (N+1 evals) and two-sided (2N evals) finite differences and forward-mode automatic differentiation.

using SymBoltz
using ForwardDiff
using LinearAlgebra
using CairoMakie

M = w0waCDM(K = nothing)
pars = merge(parameters_Planck18(M), Dict(M.X.w0 => -0.7, M.X.wa => 0.1, M.X.cₛ² => 1.0)) # keep w > -1 and wa ≠ 0 for relative steps
prob = CosmologyProblem(M, pars)
allpars = [M.g.h, M.c.Ω₀, M.b.Ω₀, M.ν.N, M.h.m_eV, M.b.YHe, M.γ.T₀, M.X.w0, M.X.wa, M.I.ns, M.I.ln_As1e10]
parlabels = [L"h", L"\Omega_{c0}", L"\Omega_{b0}", L"N_\mathrm{ur}", L"m_\nu", L"Y_\mathrm{He}", L"T_{\gamma 0}", L"w_0", L"w_a", L"n_s", L"\ln(10^{10} A_s)"]
ks = 10 .^ range(0.5, 3.5, length = 100) # in H₀/c

# Finite differences with relative step ε
δ(θ, i, ε) = ε * θ[i] * (eachindex(θ) .== i)
jacobian_onesided(f, θ, ε = 1e-2) = (f0 = f(θ); stack((f(θ + δ(θ, i, ε)) - f0) / δ(θ, i, ε)[i] for i in eachindex(θ)))
jacobian_twosided(f, θ, ε = 1e-2) = stack((f(θ + δ(θ, i, ε)) - f(θ - δ(θ, i, ε))) / 2δ(θ, i, ε)[i] for i in eachindex(θ))

# 1. Runtime as a function of the number of parameters
methods = [jacobian_onesided, jacobian_twosided, ForwardDiff.jacobian]
labels = ["One-sided finite differences", "Two-sided finite differences", "Forward-mode automatic differentiation"]
Ns = 1:length(allpars)
times = zeros(length(Ns), length(methods))
for N in Ns
    probf = remake_function(prob, allpars[1:N])
    P(θ) = spectrum_matter(probf(θ), ks)
    θ = [pars[par] for par in allpars[1:N]]
    for (j, jacobian) in enumerate(methods)
        jacobian(P, θ) # compile
        times[N, j] = minimum(@elapsed(jacobian(P, θ)) for _ in 1:2)
        println("N = $N, $(labels[j]): $(times[N, j]) s")
    end
end

fig = Figure(size = (600, 360))
axt = Axis(fig[1, 1]; xlabel = "Number of parameters N", ylabel = "Time to compute Jacobian / s", xticks = Ns)
for j in eachindex(methods)
    scatterlines!(axt, Ns, times[:, j]; label = labels[j])
end
axislegend(axt; position = :lt)
save(joinpath(@__DIR__, "..", "media", "ad_runtime.pdf"), fig)
display(fig)

# 2. Finite difference error as a function of the step size,
#    using automatic differentiation with 100x tighter solver tolerances than default as the truth
probf = remake_function(prob, allpars)
P(θ) = spectrum_matter(probf(θ), ks)
Pref(θ) = spectrum_matter(probf(θ), ks; bgreltol = 1e-9, bgabstol = 1e-9, ptreltol = 1e-7, ptabstol = 1e-7)
θ = [pars[par] for par in allpars]
J = ForwardDiff.jacobian(Pref, θ)
εs = 10 .^ range(-0.5, -5.5, step = -0.5)
relerr(Jfd) = norm.(eachcol(Jfd - J)) ./ norm.(eachcol(J)) # one per parameter
errors1 = stack(relerr(jacobian_onesided(P, θ, ε)) for ε in εs) # errors[iθ, iε]
errors2 = stack(relerr(jacobian_twosided(P, θ, ε)) for ε in εs)

fig = Figure(size = (1000, 360))
for (col, (title, errors)) in enumerate([labels[1] => errors1, labels[2] => errors2])
    ax = Axis(
        fig[1, col]; title, xlabel = "Relative step size ϵ", ylabel = col == 1 ? "Relative error to automatic differentiation" : "",
        xscale = log10, yscale = log10, xreversed = true, xticks = LogTicks(-5:-1), yticks = LogTicks(-6:2:2), yticklabelsvisible = col == 1
    )
    for i in eachindex(allpars)
        scatterlines!(ax, εs, errors[i, :]; label = parlabels[i], color = Makie.to_colormap(:glasbey_category10_n256)[i])
    end
    col == 2 && Legend(fig[1, 3], ax)
end
linkyaxes!(contents(fig[1, 1:2])...)
save(joinpath(@__DIR__, "..", "media", "ad_error.pdf"), fig)
display(fig)
