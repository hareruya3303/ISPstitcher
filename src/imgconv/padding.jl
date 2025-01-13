# Padding functions for ease of computations.

@inline function _fill_padding!(padded::AbstractArray{<:Union{AbstractFloat, RGB}}, original::AbstractArray{<:Union{AbstractFloat, RGB}}, padding::CartesianIndex)
    indices = CartesianIndices(original) .+ padding
    @inbounds padded[indices] .= original
    return nothing
end

@inline function _fill_ones!(padded::AbstractArray{<:Union{AbstractFloat, RGB}}, original::AbstractArray{<:Union{AbstractFloat, RGB}}, padding::CartesianIndex)
    indices = CartesianIndices(original) .+ padding
    @inbounds padded[indices] .= one(eltype(padded))
    return nothing
end

@inline function _fill_ones!(padded::AbstractArray{<:Union{AbstractFloat, RGB}}, indmin::Vector{<:Integer}, indmax::Vector{<:Integer})
    indices = range.(indmin, indmax)
    @inbounds padded[indices...] .= one(eltype(padded))
    return nothing
end

@inline function fill_zeros(original::AbstractArray{T}, padding_dim::N, Δpad::Integer, padval::Integer) where {T<:Union{AbstractFloat, RGB}, N<:Integer}
    dims = size(original)
    ndim = ndims(original)
    @assert padval ≥ Δpad
    padding = CartesianIndex(vcat(zeros(N, padding_dim-1), Δpad, zeros(N, ndim-padding_dim))...)
    padded_size = [dim == padding_dim ? dims[dim] + padval : dims[dim] for dim in eachindex(dims)]
    padded = typeof(original)(calloc, padded_size...)
    _fill_padding!(padded, original, padding)
    return padded
end

@inline function fill_zeros(original::AbstractArray{T}, padding_dim::Integer, Δpad::Integer, padval::Union{NTuple{N, Integer}, Vector{<:Integer}}) where {T<:Union{AbstractFloat, RGB}, N}
    dims = size(original)
    @assert padval[padding_dim] ≥ Δpad
    padding = CartesianIndex([dim == padding_dim ? Δpad : 0 for dim in eachindex(dims)]...)
    padded = typeof(original)(calloc, (size(original) .+ padval)...)
    _fill_padding!(padded, original, padding)
    return padded
end

@inline function fill_zeros(original::AbstractArray{T}, padding_dims::Union{NTuple{N0, Integer}, Vector{<:Integer}}, Δpad::Union{NTuple{N0, Integer}, Vector{<:Integer}}, padval::Union{NTuple{N1, Integer}, Vector{<:Integer}}) where {T<:Union{AbstractFloat, RGB}, N0, N1}
    dims = size(original)
    padding = CartesianIndex([dim in padding_dims ? Δpad[findfirst(isequal(dim), padding_dims)] : 0 for dim in eachindex(dims)]...)
    padded_size = [dim in padding_dims ? dims[dim] + padval[findfirst(isequal(dim), padding_dims)] : dims[dim] for dim in eachindex(dims)]
    padded = typeof(original)(calloc, padded_size...)
    _fill_padding!(padded, original, padding)
    return padded
end

@inline function fill_ones(original::AbstractArray{T}, padding_dim::N, Δpad::Integer, padval::Integer) where {T<:Union{AbstractFloat, RGB}, N<:Integer}
    dims = size(original)
    ndim = ndims(original)
    @assert padval ≥ Δpad
    padding = CartesianIndex(vcat(zeros(N, padding_dim-1), Δpad, zeros(N, ndim-padding_dim))...)
    padded_size = [dim == padding_dim ? dims[dim] + padval : dims[dim] for dim in eachindex(dims)]
    padded = typeof(original)(calloc, padded_size...)
    _fill_ones!(padded, original, padding)
    return padded
end

@inline function fill_ones(original::AbstractArray{T}, padding_dim::Integer, Δpad::Integer, padval::Union{NTuple{N, Integer}, Vector{<:Integer}}) where {T<:Union{AbstractFloat, RGB}, N}
    dims = size(original)
    println("This is the impl.")
    @assert padval[padding_dim] ≥ Δpad
    padding = CartesianIndex([dim == padding_dim ? Δpad : 0 for dim in eachindex(dims)]...)
    padded = typeof(original)(calloc, (size(original) .+ padval)...)
    _fill_ones!(padded, original, padding)
    return padded
end

@inline function fill_ones(original::AbstractArray{T}, padding_dims::Union{NTuple{N0, Integer}, Vector{<:Integer}}, Δpad::Union{NTuple{N0, Integer}, Vector{<:Integer}}, padval::Union{NTuple{N1, Integer}, Vector{<:Integer}}) where {T<:Union{AbstractFloat, RGB}, N0, N1}
    dims = size(original)
    padding = CartesianIndex([dim in padding_dims ? Δpad[findfirst(isequal(dim), padding_dims)] : 0 for dim in eachindex(dims)]...)
    padded_size = [dim in padding_dims ? dims[dim] + padval[findfirst(isequal(dim), padding_dims)] : dims[dim] for dim in eachindex(dims)]
    padded = typeof(original)(calloc, padded_size...)
    _fill_ones!(padded, original, padding)
    return padded
end

@inline function fill_ones(original::AbstractArray{T}, indmin::Union{NTuple{N, Integer}, Vector{<:Integer}}, indmax::Union{NTuple{N, Integer}, Vector{<:Integer}}) where {T<:Union{AbstractFloat, RGB}, N}
    padded = typeof(original)(calloc, size(original)...)
    _fill_ones!(padded, indmin, indmax)
    return padded
end

