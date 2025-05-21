using Plots, BenchmarkTools, Images
dir = @__DIR__
include(string(dir[1:end-4], "src/master.jl"))
using .ISPstitcher

# First step: open this lovely piece of art made by my friend, the fabulous and massively talented Axia (@axiermundi on twitter).
a0 = load(string(dir, "/Cutest_art_from_the_wonderful_Axia.jpg"))
a = a0[end-500:end, :]
b = a0[end-750:end-200, :]
c = a0[end-1100:end-600, :]
d = a0[1:600, :]
# The expected overlap is exactly 201 pixels, so that is what you should expect here too,

gcc = generalized_cross_correlation(a, b, 1)
gcc_avrg = generalized_cross_correlation(a,b,1, type = :avrg)
gcc_phat = generalized_cross_correlation(a,b,1, type = :phat)
gcc_rgb = generalized_cross_correlation(a, b, 1, rgb = :rgb)
gcc_avrg_rgb = generalized_cross_correlation(a,b,1, type = :avrg, rgb=:rgb)
gcc_phat_rgb = generalized_cross_correlation(a,b,1, type = :phat, rgb=:rgb)
#Divide the gcc_phat by its maximum just to put every criterion on the same scale.

Δ, n = setstitch([a,b,c,d], 1, silent = false)

@benchmark generalized_cross_correlation(a, b, 1)
@benchmark generalized_cross_correlation(a, b, 1, type=:avrg)
@benchmark generalized_cross_correlation(a, b, 1, type = :phat)
@benchmark generalized_cross_correlation(a, b, 1, rgb = :rgb)
@benchmark generalized_cross_correlation(a, b, 1, type=:avrg, rgb = :rgb)
@benchmark generalized_cross_correlation(a, b, 1, type = :phat, rgb = :rgb)
@benchmark setstitch([a,b,c,d], 1, silent = true)

findmax(gcc)
findmax(gcc_avrg)
findmax(gcc_phat)
findmax(gcc_rgb)
findmax(gcc_avrg_rgb)
findmax(gcc_phat_rgb)

FS = 12
p = plot(100*Array([gcc gcc_avrg gcc_phat gcc_rgb gcc_avrg_rgb gcc_phat_rgb]),
      ylim = (0, 100),
      labels = ["GCC" "GCC averaged" "GCC PHAT" "GCC (RGB)" "GCC avrg (RGB)" "GCC PHAT (RGB)"],
      legend = :outerright,
      title = "Comparison of the correlation methods",
      xlabel = "Pixels",
      ylabel = "Correlation (%)",
      xtickfontsize = FS,
      ytickfontsize = FS,
      xlabelfontsize = FS,
      ylabelfontsize = FS,)

a = a0[end-500:end, 1:1000]
b = a0[end-750:end-200, 201:end]
c = a0[end-1100:end-600, 51:end-150]
d = a0[1:600, 101:1100]

gcc_2 = generalized_cross_correlation(a, b, (1, 2))
gcc_avrg_2 = generalized_cross_correlation(a,b,(1,2), type = :avrg)
gcc_phat_2 = generalized_cross_correlation(a,b,(1,2), type = :phat)
setstitch([a,b,c,d], (1, 2), silent = false, type=:phat)

@benchmark generalized_cross_correlation(a, b, (1, 2))
@benchmark generalized_cross_correlation(a,b,(1,2), type = :avrg)
@benchmark generalized_cross_correlation(a,b,(1,2), type = :phat)
@benchmark generalized_cross_correlation(a, b, (1, 2), rgb = :rgb)
@benchmark generalized_cross_correlation(a,b,(1,2), type = :avrg, rgb = :rgb)
@benchmark generalized_cross_correlation(a,b,(1,2), type = :phat, rgb = :rgb)

savefig(p, string(dir[1:end-4], "figures/correlations.pdf"))