using SymBoltz
using CairoMakie

M = ΛCDM()
p = parameters_Planck18(M)
prob = CosmologyProblem(M, p)

# White-on-black theme for all plots
set_theme!(
    backgroundcolor = :black, textcolor = :white, linecolor = :white,
    Axis = (
        backgroundcolor = :black,
        leftspinecolor = :white, rightspinecolor = :white, topspinecolor = :white, bottomspinecolor = :white,
        xtickcolor = :white, ytickcolor = :white, xminortickcolor = :white, yminortickcolor = :white,
        xticklabelcolor = :white, yticklabelcolor = :white,
        xlabelcolor = :white, ylabelcolor = :white, titlecolor = :white,
    ),
    Lines = (color = :white, cycle = []),
    Legend = (backgroundcolor = :black, framecolor = :white, labelcolor = :white),
)

k = 10 .^ range(0.0, 3.5, length = 100)
sol = solve(prob, k)
P = spectrum_matter(sol, k)
Pnl = spectrum_matter_nonlinear(sol, k) # halofit
H0_over_c = p[M.g.h] / 2997.92458 # H₀/c in 1/Mpc

# Plot the matter power spectrum
figsize = (500, 480) # common figure size for all plots
fig = Figure(size = figsize)
ax = Axis(fig[1, 1];
    xscale = log10, yscale = log10, yticks = LogTicks(-10:10), # only integer powers of 10
    xlabel = L"k \,/\, \mathrm{Mpc}^{-1}", ylabel = L"P(k) \,/\, \mathrm{Mpc}^3",
    title = "Matter power spectrum today",
    xgridvisible = false, ygridvisible = false,
)
lines!(ax, k * H0_over_c, P / H0_over_c^3; label = "linear")
lines!(ax, k * H0_over_c, Pnl / H0_over_c^3; linestyle = :dash, label = "nonlinear (halofit)")
axislegend(ax; position = :lb)
save(joinpath(@__DIR__, "..", "media", "Pk.pdf"), fig)
fig

# CMB power spectrum
l = 2:2500
jl = SphericalBesselCache(ChebyshevIntegerInterpolator(2, 2500, 100)) # compute at Chebyshev l-nodes, then interpolate to all l
Dl = spectrum_cmb(:TT, prob, jl, l; normalization = :Dl) # dimensionless, relative to Tγ0
Dl *=(sol[M.γ.T₀] * 1e6)^2 # convert to μK²

fig = Figure(size = figsize)
ax = Axis(fig[1, 1];
    xlabel = L"l", xticks = [2, 500, 1000, 1500, 2000, 2500], ylabel = L"D_l \,/\, \mathrm{\mu K}^2",
    title = "CMB temperature power spectrum",
    xgridvisible = false, ygridvisible = false,
)
lines!(ax, l, Dl)
save(joinpath(@__DIR__, "..", "media", "Cl.pdf"), fig)
fig

