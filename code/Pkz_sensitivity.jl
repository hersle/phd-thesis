# Sensitivity of the linear matter power spectrum P(τ, k) to cosmological parameters with automatic differentiation.
# Replicates Figure 4 of https://arxiv.org/abs/2311.03291 (https://github.com/ohahn/DISCO-EB/blob/master/notebooks/paper_scripts/nb_discoeb_autodiff.ipynb),
# but at fixed conformal time τ instead of fixed a, and at fixed k in h/Mpc instead of 1/Mpc.

using SymBoltz
using ForwardDiff
using CairoMakie

M = w0waCDM()
pars = [M.g.h, M.c.Ω₀, M.b.Ω₀, M.ν.N, M.h.m_eV, M.γ.T₀, M.b.YHe, M.I.ln_As1e10, M.I.ns, M.X.w0, M.X.wa, M.X.cₛ²]
prob = CosmologyProblem(M, Dict([pars .=> NaN; M.h.N => 1])) # one massive neutrino
probf = remake_function(prob, pars)

# Differentiate with respect to DISCO-EB's parameters (h, Ωm, Ωb, Nᵤᵣ, mν, Tγ0, YHe, As, ns, w₀, 1 + wₐ, cₛ²)
labels = ["H_0", "\\Omega_m", "\\Omega_b", "N_\\mathrm{ur}", "m_\\nu", "T_{\\gamma 0}", "Y_\\mathrm{He}", "A_s", "n_s", "w_0", "(1+w_a)", "c_s^2"]
θ = [0.67742, 0.3099, 0.0488911, 2.046, 0.06, 2.7255, 0.248, 2.1064e-9, 0.96822, -0.99, 1.0, 0.99]
symboltz(θ) = [θ[1], θ[2] - θ[3], θ[3], θ[4:7]..., log(1e10 * θ[8]), θ[9:10]..., θ[11] - 1, θ[12]] # to pars

# Conformal times at scale factors from 1e-5 until today and wavenumbers in h/Mpc
sol = solve(probf(symboltz(θ)))
as = 10 .^ range(-5, 0, length = 200)
τs = SymBoltz.timeseries(sol, M.g.a, as)
ks = 10 .^ range(-4, 1, length = 100)

# CDM + baryon power spectrum P[iτ, ik]. Large scales need tighter tolerances (∝ k²):
# there the gauge-invariant overdensity ∝ (kτ)² is a tiny difference between two O(1) terms in the conformal Newtonian gauge.
tol(k) = 1e-7 * min(1, k / 1000)^2 # k in H₀/c
lnP(θ) = log.(spectrum_matter(:cb, probf(symboltz(θ)), ks ./ k100, τs; ptreltol = tol, ptabstol = tol))

# ∂ln P / ∂ln θᵢ = θᵢ ∂ln P / ∂θᵢ (also works for negative w₀)
@time J = ForwardDiff.jacobian(lnP, θ)
dlnP = reshape(J .* θ', length(τs), length(ks), length(θ)) # dlnP[iτ, ik, iθ]

# Plot like Figure 4: red = increase, blue = decrease
fig = Figure(size = (1200, 1320))
for i in eachindex(θ)
    row, col = fldmod1(i, 3)
    ax = Axis(
        fig[row, 2col-1]; xscale = log10, yscale = log10, yreversed = true, limits = (extrema(ks), extrema(as)),
        title = L"\partial \, \ln P \, / \, \partial \, \ln %$(labels[i])",
        xlabel = row == 4 ? L"k \,/\, (h/\mathrm{Mpc})" : "", ylabel = col == 1 ? L"a" : "",
    )
    c = maximum(abs, dlnP[:, :, i])
    hm = heatmap!(ax, ks, as, dlnP[:, :, i]'; colormap = Reverse(:RdBu), colorrange = (-c, +c), rasterize = 2)
    Colorbar(fig[row, 2col], hm)
end
save(joinpath(@__DIR__, "../media/Pkz_sensitivity.pdf"), fig)
display(fig)
