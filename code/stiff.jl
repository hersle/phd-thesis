# Illustrate the stiffness of the Einstein-Boltzmann equations:
# solve the perturbations for one k-mode with an implicit (Rodas5P) and an explicit (Tsit5) solver.
# The adaptive explicit solver is forced to take tiny time steps and gets nowhere before hitting maxiters,
# and the non-adaptive explicit solver diverges even with a smaller time step than the implicit solver takes.

using SymBoltz
using SymBoltz: Tsit5 # explicit solver (already loaded by SymBoltz)
using CairoMakie

M = ΛCDM()
p = parameters_Planck18(M)
prob = CosmologyProblem(M, p)

k = 1.0 # in H₀/c; the explicit solvers already fail for this (and any larger) k
maxiters = 1000 # limit number of steps, so the tiny-stepping explicit solver does not run forever
solim = solve(prob, k; ptopts = (; maxiters)).pts[1] # default implicit Rodas5P
solex = solve(prob, k; ptalg = Tsit5(), ptopts = (; maxiters)).pts[1] # adaptive explicit: stalls with tiny steps

# Fixed-step explicit with a time step smaller than the implicit solver's first (≈ 2.3e-7): diverges.
# Integrate only 200 steps, since a non-adaptive solve preallocates memory for all (tspan / dt) steps.
dt = 1e-7
ptprob = SymBoltz.setuppt(prob.pt, SymBoltz.solvebg(prob))(k) # perturbation problem for this k
ptprob = remake(ptprob; tspan = (ptprob.tspan[1], ptprob.tspan[1] + 200dt))
solfx = SymBoltz.solvept(ptprob; alg = Tsit5(), adaptive = false, dt, maxiters)

# Plot all ODE variables (one line per variable) against conformal time
τ0 = solim.t[end] # conformal time today
ymin, ymax = extrema(stack(solim.u)) # y-limits from the (correct) implicit solution
ylims = (ymin - 0.1(ymax - ymin), ymax + 0.1(ymax - ymin))
fig = Figure(size = (700, 300), fontsize = 16) # wide, for full \textwidth
xlabel = rich(rich("τ"; font = :italic), " / ", rich("τ"; font = :italic), subscript("0"))
axkw = (; xlabel, xgridvisible = false, ygridvisible = false, yticksvisible = false, yticklabelsvisible = false) # arbitrary y-units when plotting all states together
axfull = Axis(fig[1, 1]; axkw..., ylabel = "$(length(solim.u[1])) ODE states", xticks = 0:0.2:1, limits = ((-0.01, 1.01), ylims)) # small margin so points at τ = 0 and τ = τ₀ are not hidden by the axis
axzoom = Axis(fig[1, 2]; axkw..., xlabel = rich(rich("τ"; font = :italic), " / 10", superscript("−7"), " ", rich("τ"; font = :italic), subscript("0")), xticks = ((3:6) * 1e-7, string.(3:6)), limits = ((2.85e-7, 6.15e-7), ylims)) # zoom-in on the first time steps, where the explicit solvers fail
colsize!(fig.layout, 2, Relative(0.2))
sols = [
    (solim, :black, "implicit Rodas5P solver, adaptive"),
    (solfx, :blue, "explicit Tsit5 solver, non-adaptive"),
    (solex, :red, "explicit Tsit5 solver, adaptive"), # last, so it is on top
]
for ax in (axfull, axzoom), (sol, color, name) in sols
    for i in eachindex(sol.u[1])
        y = clamp.(sol[i, :], -1e5, 1e5) # far outside the plot, but CairoMakie fails to draw lines to huge values (~1e27)
        scatterlines!(ax, sol.t / τ0, y; color, linewidth = 1.5, marker = :vline, markersize = 15, label = "$name ($(length(sol.t)) steps, $(sol.retcode))")
    end
end
axislegend(axfull; position = :lt, unique = true, framevisible = false)
save(joinpath(@__DIR__, "..", "media", "stiff.pdf"), fig)
fig
