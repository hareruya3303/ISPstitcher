# These are routines meant to set up the stitch dimensions based on one or a series of images.
# The arguments here are the same arguments used for calculating the correlations, as that is how the stitch dimensions are determined.

function _setstitch_silent(gcc::AbstractVector{<:Real}, n1::NTuple{N, Integer}, n2::NTuple{N, Integer}, dim::Integer = 1) where {N}
    _, Δ = findmax(gcc)
    n = n1[dim] + n2[dim] - Δ
    return Δ, n
end
 
function _setstitch_loud(gcc::AbstractVector{<:Real}, n1::NTuple{N, Integer}, n2::NTuple{N, Integer}, dim::Integer = 1) where {N}
    p, Δ = findmax(gcc)
    println("The correlation is ", p, " with a total of ", Δ," pixels.")
    n = n1[dim] + n2[dim] - Δ
    return Δ, n
end

function setstitch(a::AbstractMatrix{C}, b::AbstractMatrix{C}, dim::Integer=1; type::Symbol=:Averaged, gpu = false, rgb::Symbol = :Gray, silent=true) where {C<:RGB}
    gcc = generalized_cross_correlation(a, b, dim, type = type, gpu=gpu, rgb=rgb)
    na = size(a)
    nb = size(b)
    Δ, n = silent ? _setstitch_silent(gcc, na, nb) : _setstitch_loud(gcc, na, nb)
    return Δ, n
end

function setstitch(a::AbstractVector{<:AbstractMatrix{C}}, dim::Integer=1; type::Symbol=:Averaged, gpu = false, rgb::Symbol = :Gray, silent=true) where {C<:RGB}
    L = length(a)
    Δ = Vector{Integer}(calloc, L-1)
    n = Vector{Integer}(calloc, L)
    n[1] = size(a[1], dim)
    if silent
        @showprogress for t in 1:L-1
            gcc = generalized_cross_correlation(a[t], a[t+1], dim, type = type, gpu=gpu, rgb=rgb)
            na = n[t]
            nb = size(a[t+1], dim)
            Δ[t], n[t+1] = _setstitch_silent(gcc, na, nb)
        end
    else
        for t in 1:L-1
            gcc = generalized_cross_correlation(a[t], a[t+1], dim, type = type, gpu=gpu, rgb=rgb)
            na = n[t]
            nb = size(a[t+1], dim)
            Δ[t], n[t+1] = _setstitch_loud(gcc, na, nb)
        end
    end
    Δ, n
end

function _setshift(n1::Integer, n2::Integer, δ::Integer)
    n_max = max(n1, n2)
    Δ = δ ≤ n_max/2 ? δ-1 : δ-(n_max+1)
    return Δ
end

function _setstitch(n1::Union{NTuple{N0, Integer}, Vector{Integer}}, n2::Union{NTuple{N0, Integer}, Vector{Integer}}, dims::Union{NTuple{N, Integer}, Vector{Integer}}, δ::Union{NTuple{N, Integer}, Vector{Integer}}) where {N0, N}
    Δ = Vector{Integer}(calloc, N)
    Δ[1] = δ[1]
    n = n1[dims[1]] + n2[dims[1]] - δ[1]
    for d = 2:N
        Δ[d]

end

function setstitch(a::AbstractMatrix{C}, b::AbstractMatrix{C}, dim::NTuple{N, Integer}; type::Symbol=:Averaged, gpu = false, rgb::Symbol = :Gray, silent=true) where {C<:RGB, N}
    gcc = generalized_cross_correlation(a, b, dim, type = type, gpu=gpu, rgb=rgb)
    na = size(a)
    nb = size(b)
    Δ, n = silent ? _setstitch_silent(gcc, na, nb, dim) : _setstitch_loud(gcc, na, nb, dim)
    return Δ, n
end

function setstich(a::AbstractVector{AbstractMatrix{C}}, dim::NTuple{N, Integer}; type::Symbol=:Averaged, gpu = false, rgb::Symbol = :Gray, silent=true) where {C<:RGB, N} 
    na = size(a[1])
    l = length(a)
    Δ = Matrix{Integer}(calloc, l-1, N)
    
    