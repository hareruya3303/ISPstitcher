# Accumulations without averages.

# General case on singleton integer dimension. assumes all dimensions are equal, but I do no checking. You are on your own.
# That said, these inplace routines are not meant to be used directly by anyone, they have pre-processing in the exported routines that only invokes them on same-sized arrays.
# It would be easy enough to support different-sized dimensions in general, but that is not a use-case I am interested in for now.
function _cross_correlation!(accumulator::AbstractArray{<:AbstractFloat}, a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, dim::Integer) 
    n = size(a)
    ndim = ndims(a)
    l = size(a, dim)
    all_dims = 1:ndims(a)
    a_aug = fill_zeros(a,  dim, l-1, l-1)
    b_aug = fill_zeros(b,  dim,   0, l-1)
    mean_dims = all_dims[all_dims .!= dim]
    σa = fill_zeros(sum(abs2, a, dims = mean_dims), dim, l-1, l-1)
    σb = fill_zeros(sum(abs2, b, dims = mean_dims), dim, 0, l-1)
    dim_min = ones(Int, ndim)
    dim_max = [d == dim ? l : 1 for d in 1:ndim]
    a_pad = fill_ones(σa, dim_min, dim_max)
    b_pad = reverse(a_pad, dims=dim)
    accumulate!(σa, a_pad, dim)
    σb = accumulate(b_pad, σb, dim)
    σab = accumulate(a_aug, b_aug, dim, mean_dims)
    ind_mean = [d == dim ? (1:l) : 1:size(σa, d) for d in all_dims]
    @inbounds σab[ind_mean...] ./= sqrt.(σa[ind_mean...] .* σb[ind_mean...])
    @inbounds accumulator .= dropdims(σab[ind_mean...], dims = Tuple(mean_dims))
    return nothing
end

function _cross_correlation!(accumulator::AbstractVector{<:AbstractFloat}, a::AbstractVector{<:AbstractFloat}, b::AbstractVector{<:AbstractFloat})
    la = length(a)
    lb = length(b)
    l = min(la, lb)
    a_aug = fill_zeros(a,  1, lb-1, lb-1)
    b_aug = fill_zeros(b,  1,   0, la-1)
    a_pad = fill_ones(b, 1,   0, la-1)
    b_pad = fill_ones(a, 1, lb-1, lb-1)
    σa = similar(a_aug)
    σb = similar(b_aug)
    _accumulate!(σa, a_aug.^2, a_pad)
    _accumulate!(σb, b_aug.^2, b_pad)
    Accumulator = similar(a_aug)
    _accumulate!(Accumulator, a_aug, b_aug)
    @inbounds accumulator .= (Accumulator[1:l])./sqrt(σa[1:l] .* σb[1:l])
    return nothing
end

# Best case scenario for stitch with shift: images are the same size.
function _cross_correlation_!(accumulator::AbstractArray{<:AbstractFloat}, a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}})
    na = size(a)
    nb = size(b)
    all_dims = 1:ndims(a)
    # @assert na[dims[1]] == nb[dims[1]]
    # @assert na[.!(in.(all_dims, [dims]))] == nb[.!(in.(all_dims, [dims]))]
    l = na[dims[1]]
    excluded_dims = all_dims[.!(in.(all_dims, [dims]))]
    mean_dims = vcat(excluded_dims, dims[2:end]...)
    a_aug = fill_zeros(a, dims[1], l-1, l-1)
    b_aug = fill_zeros(b, dims[1],   0, l-1)
    σa = fill_zeros(sum(abs2, a, dims = mean_dims), dims[1], l-1, l-1)
    σb = fill_zeros(sum(abs2, b, dims = mean_dims), dims[1], 0, l-1)
    a_dim_min, a_dim_max, b_dim_min, b_dim_max, fft_dims_a, fft_dims_b = get_pad_indices_for_ones(na, nb, dims[1])
    a_pad = fill_ones(σa, a_dim_min, a_dim_max)
    b_pad = fill_ones(σb, b_dim_min, b_dim_max)
    accumulate!(σa, a_pad, fft_dims_a)
    σb = accumulate(b_pad, σb, fft_dims_b)
    reverse!(σb, dims=dims[2:end])
    σab = accumulate(a_aug, b_aug, dims, excluded_dims)
    ind_mean = [d == dims[1] ? (1:l) : 1:size(σa, d) for d in all_dims]
    ind_ab = [d == dims[1] ? (1:l) : d in dims ? (1:max(na[d], nb[d])) : 1:1 for d in all_dims]
    @inbounds σab[ind_ab...] ./= sqrt.(σa[ind_mean...] .* σb[ind_mean...])
    accumulator .= dropdims(σab[ind_ab...], dims=Tuple(excluded_dims))
    return nothing
end

# Close to the worst case scenario: cross correlation when the arras have different dimensions outside of the lead dimension.
# The dimensions are MEANT to be different. The first dimension in dims here is what I call the leading dimension. It is the dimension where the inputs MUST be the same size.
# It is far too annoying, albeit not impossible, to rewrite this to accept all manner of madness such as different dimensions altogether and stuff. Maybe sometime in the future, but for the use case of stitching anime screenshots it would be even more overkill than what I've already written.
function _cross_correlation__!(accumulator::AbstractArray{<:AbstractFloat}, a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}})
    na = size(a)
    nb = size(b)
    all_dims = 1:ndims(a)
    # @assert na[dims[1]] == nb[dims[1]]
    # @assert na[.!(in.(all_dims, [dims]))] == nb[.!(in.(all_dims, [dims]))]
    l = na[dims[1]]
    pad_dims_a, pad_dims_b, Δpad_a, Δpad_b, padval_a, padval_b, excluded_dims = _determine_padding_conv(na, nb, dims[1])
    excluded_dims_a = vcat(excluded_dims, pad_dims_a[pad_dims_a .!= dims[1]])
    excluded_dims_b = vcat(excluded_dims, pad_dims_b[pad_dims_b .!= dims[1]])
    a_aug = fill_zeros(a, pad_dims_a, Δpad_a, padval_a)
    b_aug = fill_zeros(b, pad_dims_b, Δpad_b, padval_b)
    σa = fill_zeros(sum(abs2, a, dims = excluded_dims_a), dims[1], l-1, l-1)
    σb = fill_zeros(sum(abs2, b, dims = excluded_dims_b), dims[1], 0, l-1)
    a_dim_min, a_dim_max, b_dim_min, b_dim_max, fft_dims_a, fft_dims_b = get_pad_indices_for_ones(na, nb, dims[1])
    a_pad = fill_ones(σa, a_dim_min, a_dim_max)
    b_pad = fill_ones(σb, b_dim_min, b_dim_max)
    accumulate!(σa, a_pad, fft_dims_a)
    σb = accumulate(b_pad, σb, fft_dims_b)
    reverse!(σb, dims=dims[2:end])
    fft_dims = all_dims[in.(all_dims, [fft_dims_a]) .|| in.(all_dims, [fft_dims_b])]
    σab = accumulate(a_aug, b_aug, fft_dims, excluded_dims)
    ind_a = [d == dims[1] ? (1:l) : 1:size(σa, d) for d in all_dims]
    ind_b = [d == dims[1] ? (1:l) : 1:size(σb, d) for d in all_dims]
    ind_ab = [d == dims[1] ? (1:l) : d in dims ? (1:max(na[d], nb[d])) : 1:1 for d in all_dims]
    @inbounds σab[ind_ab...] ./= sqrt.(σa[ind_a...] .* σb[ind_b...])
    accumulator .= dropdims(σab[ind_ab...], dims=Tuple(excluded_dims))
    return nothing
end

function _cross_correlation!(accumulator::AbstractArray{<:AbstractFloat}, a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}})
    na = size(a)
    nb = size(b)
    @assert(na[dims[1]] == nb[dims[1]], "Inputs must have same size on the primary correlation dimension!")
    if na == nb
        _cross_correlation_!(accumulator, a, b, dims)
    else
        _cross_correlation__!(accumulator, a, b, dims)
    end
    return nothing
end

# Finally, the not in-place implementations, which allocate the necessary arrays of the type desired.

function _cross_correlation(a::AbstractArray{<:AbstractFloat, N}, b::AbstractArray{<:AbstractFloat, N}, dim::Integer) where {N}
    na = size(a)
    nb = size(b)
    l = min(na[dim], nb[dim])
    all_dims = 1:N
    @assert(na[all_dims .!= dim] == nb[all_dims .!= dim], "Cross-correlation on singleton dimensions requires arrays equally sized on the other dimensions!")
    accumulator = similar(a, l)
    _cross_correlation!(accumulator, a, b, dim)
    return accumulator
end

function _cross_correlation(a::AbstractArray{<:AbstractFloat, N}, b::AbstractArray{<:AbstractFloat, N}, dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}}) where {N}
    na = size(a)
    nb = size(b)
    l = min(na[dims[1]], nb[dims[1]])
    accumulator = similar(a, [d == dims[1] ? l : max(na[d], nb[d]) for d in dims]...)
    _cross_correlation!(accumulator, a, b, dims)
    return accumulator
end