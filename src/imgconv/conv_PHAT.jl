# Correlations calculated with PHAT phase transformations.

# Cross correlation calculated with the phase transformation, meaning the frequency domain is normalized. In this case, there is no point or meaning to subtracting the mean, or normalizing in the spatial domain.

# GCC PHAT is a bad idea for the basic case 
function _cross_correlation_PHAT!(accumulator::AbstractArray{<:AbstractFloat}, a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, dim::Integer) 
    l = size(a, dim)
    all_dims = 1:ndims(a)
    excluded_dims = (all_dims[(all_dims .!= dim)])
    a_aug = fill_zeros(a,  dim, l-1, l-1)
    b_aug = fill_zeros(b,  dim,   0, l-1)
    σab = accumulate_PHAT(a_aug, b_aug, dim, excluded_dims)
    ind_mean = [d == dim ? (1:l) : 1:size(σab, d) for d in all_dims]
    @inbounds accumulator .= dropdims(σab[ind_mean...], dims = Tuple(excluded_dims))
    return nothing
end

function _cross_correlation_PHAT!(accumulator::AbstractVector{<:AbstractFloat}, a::AbstractVector{<:AbstractFloat}, b::AbstractVector{<:AbstractFloat})
    la = length(a)
    lb = length(b)
    l = min(la, lb)
    a_aug = fill_zeros(a,  1, lb-1, lb-1)
    b_aug = fill_zeros(b,  1,   0, la-1)
    Accumulator = similar(a_aug)
    _accumulate_PHAT!(Accumulator, a_aug, b_aug)
    @inbounds accumulator .= Accumulator[1:l]
    return nothing
end

# Best case scenario for stitching shots with shift. It is much less significant than for the other two cases.
function _cross_correlation_PHAT_!(accumulator::AbstractArray{<:AbstractFloat}, a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}})
    na = size(a)
    nb = size(b)
    all_dims = 1:ndims(a)
    l = na[dims[1]]
    excluded_dims = all_dims[.!(in.(all_dims, [dims]))]
    a_aug = fill_zeros(a, dims[1], l-1, l-1)
    b_aug = fill_zeros(b, dims[1],   0, l-1)
    σab = accumulate_PHAT(a_aug, b_aug, dims, excluded_dims)
    ind_ab = [d == dims[1] ? (1:l) : d in dims ? (1:max(na[d], nb[d])) : 1:1 for d in all_dims]
    @inbounds accumulator .= dropdims(σab[ind_ab...], dims=Tuple(excluded_dims))
    return nothing
end

# Worst case scenario for the PHAT, but best case scenario for the stitch with shift with different dimensions. Basically, the use of PHAT makes it so the system only requires 1 convolution, as opposed to the basic case which requires at least 3 and the averaged stitch, which requires at least 5.
function _cross_correlation_PHAT__!(accumulator::AbstractArray{<:AbstractFloat}, a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}})
    na = size(a)
    nb = size(b)
    all_dims = 1:ndims(a)
    pad_dims_a, pad_dims_b, Δpad_a, Δpad_b, padval_a, padval_b, excluded_dims = _determine_padding_conv(na, nb, dims[1])
    l = na[dims[1]]
    a_aug = fill_zeros(a, pad_dims_a, Δpad_a, padval_a)
    b_aug = fill_zeros(b, pad_dims_b, Δpad_b, padval_b)
    σab = similar(a_aug)
    σab = accumulate_PHAT(a_aug, b_aug, dims, excluded_dims)
    ind_ab = [d == dims[1] ? (1:l) : d in dims ? (1:max(na[d], nb[d])) : 1:1 for d in all_dims]
    @inbounds accumulator .= dropdims(σab[ind_ab...], dims=Tuple(excluded_dims))
    return nothing
end

function _cross_correlation_PHAT!(accumulator::AbstractArray{<:AbstractFloat}, a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}})
    na = size(a)
    nb = size(b)
    @assert(na[dims[1]] == nb[dims[1]], "Inputs must have same size on the primary correlation dimension!")
    if na == nb
        _cross_correlation_PHAT_!(accumulator, a, b, dims)
    else
        _cross_correlation_PHAT__!(accumulator, a, b, dims)
    end
    return nothing
end

function _cross_correlation_PHAT(a::AbstractArray{<:AbstractFloat, N}, b::AbstractArray{<:AbstractFloat, N}, dim::Integer) where {N}
    na = size(a)
    nb = size(b)
    l = min(na[dim], nb[dim])
    all_dims = 1:N
    @assert(na[all_dims .!= dim] == nb[all_dims .!= dim], "Cross-correlation on singleton dimensions requires arrays equally sized on the other dimensions!")
    accumulator = similar(a, l)
    _cross_correlation_PHAT!(accumulator, a, b, dim)
    return accumulator
end

function _cross_correlation_PHAT(a::AbstractArray{<:AbstractFloat, N}, b::AbstractArray{<:AbstractFloat, N}, dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}}) where {N}
    na = size(a)
    nb = size(b)
    l = min(na[dims[1]], nb[dims[1]])
    accumulator = similar(a, [d == dims[1] ? l : max(na[d], nb[d]) for d in dims]...)
    _cross_correlation_PHAT!(accumulator, a, b, dims)
    return accumulator
end