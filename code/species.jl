using SymBoltz
using CairoMakie

M = ΛCDM()
p = parameters_Planck18(M)
prob = CosmologyProblem(M, p)

sol = solve(prob) # background only
a = sol[M.g.a]
Ω = stack(sol[[M.γ.ρ, (M.ν.ρ + M.h.ρ), M.b.ρ, M.c.ρ, M.Λ.ρ] ./ M.G.ρ]) # energy fractions; one row per species, stacked bottom to top
Ωcum = [zeros(1, length(a)); cumsum(Ω; dims = 1)] # cumulative fractions (band edges, from 0 to 1) <3

labels = ["photons", "neutrinos", "baryons", "dark matter", "dark energy"]
colors = Makie.wong_colors()[[3, 4, 1, 2, 6]] # skip light blue, too similar to blue
alabels = [10^-5.8, 10^-5.8,10^-1.5, 10^-2, 10^-0.13] # scale factors at which to label each band
ylabels = [nothing, nothing, nothing, nothing, 0.816] # nothing = middle of band
rotations = [0, 0, 0, 0, π/2] # dark energy band is narrow, so make the label vertical

# Plot the energy fractions as stacked bands (a "pie chart" at every a)
fig = Figure(size = (700, 350), fontsize = 16, figure_padding = (2, 12, 2, 2)) # (left, right, bottom, top); right leaves room for the 10⁰ tick label # wide, for full \textwidth
ax = Axis(fig[1, 1];
    xscale = log10, xticks = LogTicks(-10:0),
    yticks = (0:0.2:1, ["0%", "20%", "40%", "60%", "80%", "100%"]),
    xlabel = rich("scale factor ", rich("a"; font = :italic)), ylabel = "fraction of energy",
    title = "Energy content of an expanding ΛCDM universe",
    xgridvisible = false, ygridvisible = false,
    limits = ((1e-7, 1), (0, 1)),
)
for i in eachindex(labels)
    band!(ax, a, Ωcum[i, :], Ωcum[i+1, :]; color = colors[i])
    j = argmin(abs.(log.(a ./ alabels[i])))
    y = something(ylabels[i], (Ωcum[i, j] + Ωcum[i+1, j]) / 2)
    text!(ax, alabels[i], y; text = labels[i], align = (:center, :center), rotation = rotations[i], color = :black)
end
save(joinpath(@__DIR__, "..", "media", "species.pdf"), fig)
fig
