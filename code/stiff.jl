# Illustrate the stiffness of the Einstein-Boltzmann equations:
# solve the perturbations for one k-mode with an implicit (Rodas5P) and an explicit (Tsit5) solver.
# The explicit solver is forced to take tiny time steps and gets nowhere before hitting maxiters.

using SymBoltz
using SymBoltz: Tsit5 # explicit solver (already loaded by SymBoltz)
using CairoMakie

M = ΛCDM()
p = parameters_Planck18(M)
prob = CosmologyProblem(M, p)

k = 1.0 # in H₀/c; the explicit solver already stalls for this (and any larger) k
maxiters = 1000 # limit number of steps, so the tiny-stepping explicit solver does not run forever
solim = solve(prob, k; ptopts = (; maxiters)).pts[1] # default implicit Rodas5P
solex = solve(prob, k; ptalg = Tsit5(), ptopts = (; maxiters)).pts[1] # adaptive explicit: stalls with tiny steps
solfx = solve(prob, k; ptalg = Tsit5(), ptopts = (; maxiters, adaptive = false, dt = 1e-3)).pts[1] # fixed-step explicit: diverges

# Plot all ODE variables (one line per variable) against conformal time
τ0 = solim.t[end] # conformal time today
ymin, ymax = extrema(stack(solim.u)) # y-limits from the (correct) implicit solution
Δy = 0.05 * (ymax - ymin)
fig = Figure(size = (700, 300), fontsize = 16) # wide, for full \textwidth
ax = Axis(fig[1, 1];
    xlabel = rich(rich("τ"; font = :italic), " / ", rich("τ"; font = :italic), subscript("0")), ylabel = "$(length(solim.u[1])) ODE states",
    xticks = 0:0.2:1, xgridvisible = false, ygridvisible = false,
    yticksvisible = false, yticklabelsvisible = false, # arbitrary units when plotting all states together
    limits = ((-0.01, 1.01), (ymin - Δy, ymax + Δy)), # small margin so points at τ = 0 and τ = τ₀ are not hidden by the axis
)
for (sol, color, name) in [(solim, :black, "implicit Rodas5P solver, adaptive"), (solfx, :blue, "explicit Tsit5 solver, non-adaptive"), (solex, :red, "explicit Tsit5 solver, adaptive")] # explicit on top
    for i in eachindex(sol.u[1])
        scatterlines!(ax, sol.t / τ0, sol[i, :]; color, linewidth = 1.5, marker = :vline, markersize = 15, label = "$name ($(length(sol.t)) steps, $(sol.retcode))")
    end
end
axislegend(ax; position = :lt, unique = true, framevisible = false)
save(joinpath(@__DIR__, "..", "media", "stiff.pdf"), fig)
fig
