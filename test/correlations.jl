using Plots, BenchmarkTools, Images
dir = @__DIR__
include(string(dir[1:end-4], "src/master.jl"))
using .ISPstitcher

# First step: open this lovely piece of art made by my friend, the fabulous and massively talented Axia (@axiermundi).
a0 = load(string(dir, "/Cutest_art_from_cutest_artist.jpg"))
a = a0[800:end, :]
b = a0[1:1000, :]
# The expected overlap is exactly 201 pixels, so that is what you should expect here too,

gcc = generalized_cross_correlation(a, b, 1)
gcc_avrg = generalized_cross_correlation(a,b,1, type = :avrg)
gcc_phat = generalized_cross_correlation(a,b,1, type = :phat)
#Divide the gcc_phat by its maximum just to put every criterion on the same scale.
gcc_phat /= maximum(gcc_phat)

findmax(gcc)
findmax(gcc_avrg)
findmax(gcc_phat)

FS = 12
plot(100*[gcc gcc_avrg gcc_phat],
      ylim = (0, 100),
      labels = ["GCC no average" "GCC averaged" "GCC PHAT"],
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

surface(gcc_2)
surface(gcc_avrg_2)
surface(gcc_phat_2)