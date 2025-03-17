using Plots, BenchmarkTools, Images
dir = @__DIR__
include(string(dir[1:end-4], "src/master.jl"))
using .ISPstitcher

# First step: open this lovely piece of art made by my friend, the fabulous and massively talented Axia (@axiermundi on twitter).
a0 = load(string(dir, "/Cutest_art_from_the_wonderful_Axia.jpg"))
a = a0[end-1439:end, :]
b = a0[1:1439, :]
# The expected overlap is exactly 201 pixels, so that is what you should expect here too,

gcc = generalized_cross_correlation(a, b, 1)
gcc_avrg = generalized_cross_correlation(a,b,1, type = :avrg)
gcc_phat = generalized_cross_correlation(a,b,1, type = :phat)
gcc_rgb = generalized_cross_correlation(a, b, 1, rgb = :rgb)
gcc_avrg_rgb = generalized_cross_correlation(a,b,1, type = :avrg, rgb=:rgb)
gcc_phat_rgb = generalized_cross_correlation(a,b,1, type = :phat, rgb=:rgb)
#Divide the gcc_phat by its maximum just to put every criterion on the same scale.
gcc_phat /= maximum(gcc_phat)
gcc_phat_rgb /= maximum(gcc_phat_rgb)

@benchmark generalized_cross_correlation(a, b, 1)
@benchmark generalized_cross_correlation(a, b, 1, type=:avrg)
@benchmark generalized_cross_correlation(a, b, 1, type = :phat)
@benchmark generalized_cross_correlation(a, b, 1, rgb = :rgb)
@benchmark generalized_cross_correlation(a, b, 1, type=:avrg, rgb = :rgb)
@benchmark generalized_cross_correlation(a, b, 1, type = :phat, rgb = :rgb)

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

gcc_2 = generalized_cross_correlation(a, b, (1, 2))
gcc_avrg_2 = generalized_cross_correlation(a,b,(1,2), type = :avrg)
gcc_phat_2 = generalized_cross_correlation(a,b,(1,2), type = :phat)
#Divide the gcc_phat by its maximum just to put every criterion on the same scale.
gcc_phat_2 /= maximum(gcc_phat_2)

findmax(gcc_2)
findmax(gcc_avrg_2)
findmax(gcc_phat_2)


@benchmark generalized_cross_correlation(a, b, (1, 2))
@benchmark generalized_cross_correlation(a,b,(1,2), type = :avrg)
@benchmark generalized_cross_correlation(a,b,(1,2), type = :phat)
@benchmark generalized_cross_correlation(a, b, (1, 2), rgb = :rgb)
@benchmark generalized_cross_correlation(a,b,(1,2), type = :avrg, rgb = :rgb)
@benchmark generalized_cross_correlation(a,b,(1,2), type = :phat, rgb = :rgb)

surface(gcc_2)
surface(gcc_avrg_2)
surface(gcc_phat_2)

savefig(p, string(dir[1:end-4], "figures/correlations.pdf"))