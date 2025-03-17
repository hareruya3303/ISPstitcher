module ISPstitcher
dir = @__DIR__
using FFTW, Images, LinearAlgebra, Statistics, DSP, Suppressor, ArrayAllocators, Match, OffsetArrays
using FFTW: rFFTWPlan
import Base: zeros

GPUArray, GPUVector, GPUMatrix = try begin using CUDA
        @suppress_out CUDA.device()
        println("\nCUDA GPU detected! GPU option for acceleration will default to this! Set gpu=true option to use it!")
    end
    CuArray, CuVector, CuMatrix
catch e
    try begin using Metal
            @suppress_out Metal.versioninfo()
            println("\nMetal GPU detected! Unfortunately, nothing is set up to work with it yet!")
        end
        MtlArray, MtlVector, MtlMatrix
    catch em
        Array, Vector, Matrix
    end
end;

include(string(dir, "/imgconv/imgconv.jl"))
end
