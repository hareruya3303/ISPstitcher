# These are routines meant to set up the stitch dimensions based on one or a series of images.
# The arguments here are the same arguments used for calculating the correlations, as that is how the stitch dimensions are determined.

@inline function _setstitch_silent(gcc::AbstractVector{<:Real}, n1::Integer, n2::Integer)
    _, Δ = findmax(gcc)
    n = n1 + n2 - Δ
    return Δ, n
end
 
@inline function _setstitch_loud(gcc::AbstractVector{<:Real}, n1::Integer, n2::Integer)
    p, Δ = findmax(gcc)
    println("The correlation is ", p, " with a total of ", Δ," pixels.")
    n = n1 + n2 - Δ
    return Δ, n
end

@inline function setstitch(a::AbstractMatrix{C}, b::AbstractMatrix{C}, dim::Integer=1; type::Symbol=:Averaged, gpu = false, rgb::Symbol = :Gray, silent=true) where {C<:RGB}
    gcc = generalized_cross_correlation(a, b, dim, type = type, gpu=gpu, rgb=rgb)
    na = size(a)
    nb = size(b)
    Δ, n = silent ? _setstitch_silent(gcc, na, nb) : _setstitch_loud(gcc, na, nb)
    return Δ, n
end

@inline function setstitch(a::Union{AbstractVector{<:AbstractMatrix{C}}, NTuple{N, <:AbstractMatrix{<:C}}}, dim::Integer=1; type::Symbol=:Averaged, gpu = false, rgb::Symbol = :Gray, silent=true) where {C<:RGB, N}
    L = length(a)
    Δ = Vector{Int}(calloc, L-1)
    n = Vector{Int}(calloc, L)
    n[1] = size(a[1], dim)
    if silent
        @showprogress for t in 1:L-1
            gcc = generalized_cross_correlation(a[t], a[t+1], dim, type = type, gpu=gpu, rgb=rgb)
            na = n[t]
            nb = size(a[t+1], dim)
            Δ[t], n[t+1] = @inline _setstitch_silent(gcc, na, nb)
        end
    else
        for t in 1:L-1
            gcc = generalized_cross_correlation(a[t], a[t+1], dim, type = type, gpu=gpu, rgb=rgb)
            na = n[t]
            nb = size(a[t+1], dim)
            Δ[t], n[t+1] = @inline _setstitch_loud(gcc, na, nb)
        end
    end
    Δ, n
end

@inline function _setstitch_silent(gcc::AbstractMatrix{<:Real}, n1::Integer, n2::Integer, dims::NTuple{N, Integer} = 1) where {N}
    _, Δ = findmax(gcc)
    n = n1 + n2 - Δ[dims[1]]
    return Δ[dims[1]], n, Δ[dims[2]]
end

@inline function _setstitch_loud(gcc::AbstractMatrix{<:Real}, n1::Integer, n2::Integer, dims::NTuple{N, Integer} = 1) where {N}
    p, Δ = findmax(gcc)
    println("The correlation is ", p, " with a total of ", Δ[dims[1]]," pixels of overlap and a shift of ", Δ[dims[2]]," pixels.")
    n = n1 + n2 - Δ[dims[1]]
    return Δ[dims[1]], n, Δ[dims[2]]
end

@inline function setstitch(a::AbstractMatrix{C}, b::AbstractMatrix{C}, dims::NTuple{2, Integer}; type::Symbol=:Averaged, gpu = false, rgb::Symbol = :Gray, silent=true) where {C<:RGB}
    dim = dims[1]
    gcc = generalized_cross_correlation(a, b, dims, type = type, gpu=gpu, rgb=rgb)
    na = size(a, dim)
    nb = size(b, dim)
    Δ, n = silent ? _setstitch_silent(gcc, na, nb, dim) : _setstitch_loud(gcc, na, nb, dim)
    return Δ[dim], n, Δ[dims[2]]
end

@inline function setstitch(A::Union{AbstractVector{<:AbstractMatrix{C}}, NTuple{N, AbstractMatrix{C}}}, dims::NTuple{2, Integer}; type::Symbol=:Averaged, gpu = false, rgb::Symbol = :Gray, silent=true) where {C<:RGB, N}
    L = length(A)
    Δ = Vector{Int}(calloc, L-1)
    n_left = Vector{Int}(calloc, L)
    n_right = Vector{Int}(calloc, L)
    n = Vector{Int}(calloc, L)
    n[1] = size(A[1], dims[1])
    n_right[1] = size(A[1], dims[2])
    if silent
        @showprogress for t in 1:L-1
            gcc = generalized_cross_correlation(A[t], A[t+1], dims, type = type, gpu=gpu, rgb=rgb)
            na = n[t]
            nb = size(A[t+1], dims[1])
            @inbounds n_right[t+1] = size(A[t+1], dims[2])
            @inbounds Δ[t], n[t+1], delta = @inline _setstitch_silent(gcc, na, nb, dims)
            if delta ≥ 0
                @inbounds n_left[t+1] = n_left[t] + delta
            else
                @inbounds n_left[t+1] = n_left[t]
                n_left[1:t] .-= delta
            end
        end
    else
        for t in 1:L-1
            gcc = generalized_cross_correlation(A[t], A[t+1], dims, type = type, gpu=gpu, rgb=rgb)
            na = n[t]
            nb = size(A[t+1], dims[1])
            @inbounds n_right[t+1] = size(A[t+1], dims[2])
            @inbounds Δ[t], n[t+1], delta = @inline _setstitch_loud(gcc, na, nb, dims)
            if delta ≥ 0
                @inbounds n_left[t+1] = n_left[t] + delta
            else
                @inbounds n_left[t+1] = n_left[t]
                n_left[1:t] .-= delta
            end
        end
    end
    n_left .-= minimum(n_left)
    @inbounds n_right .+= n_left
    return Δ, n, n_left, n_right
end

