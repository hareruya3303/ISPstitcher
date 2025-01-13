## Accumulations with averages.

# General case on singleton integer dimension. assumes all dimensions are equal, but I do no checking. You are on your own.
# That said, these inplace routines are not meant to be used directly by anyone, they have pre-processing in the exported routines that only invokes them on same-sized arrays.
# It would be easy enough to support different-sized dimensions in general, but that is not a use-case I am interested in for now.
function _cross_correlation_averaged!(accumulator::AbstractArray{<:AbstractFloat}, a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, dim::Integer) 
    n = size(a)
    l = size(a, dim)
    all_dims = 1:ndims(a)
    a_aug = fill_zeros(a,  dim, l-1, l-1)
    b_aug = fill_zeros(b,  dim,   0, l-1)
    mean_dims = all_dims[all_dims .!= dim]
    μa = fill_zeros(sum(a, dims = mean_dims), dim, l-1, l-1)
    μb = fill_zeros(sum(b, dims = mean_dims), dim, 0, l-1)
    σa = fill_zeros(sum(abs2, a, dims = mean_dims), dim, l-1, l-1)
    σb = fill_zeros(sum(abs2, b, dims = mean_dims), dim, 0, l-1)
    a_dim_min, a_dim_max, b_dim_min, b_dim_max, fft_dims_a, fft_dims_b = get_pad_indices_for_ones(n, n, dim)
    a_pad = fill_ones(μa, a_dim_min, a_dim_max)
    b_pad = reverse(a_pad, dims=dim)
    accumulate!(μa, a_pad, dim)
    μb = accumulate(b_pad, μb, dim)
    accumulate!(σa, a_pad, dim)
    σb = accumulate(b_pad, σb, dim)
    σab = accumulate(a_aug, b_aug, dim, mean_dims)
    ind_mean = [d == dim ? (1:l) : 1:size(μa, d) for d in all_dims]
    ls = typeof(μa)(calloc, [d == dim ? l : 1 for d in all_dims]...)
    @inbounds ls[:] .= prod(n)*(1:l)/l
    @inbounds σa[ind_mean...] .-=  abs2.(μa[ind_mean...]) ./ ls
    @inbounds σb[ind_mean...] .-=  abs2.(μb[ind_mean...]) ./ ls
    @inbounds σab[ind_mean...] .-= (μa[ind_mean...] .* μb[ind_mean...]) ./ls
    @inbounds σab[ind_mean...] ./= sqrt.(σa[ind_mean...] .* σb[ind_mean...])
    @inbounds accumulator .= dropdims(σab[ind_mean...], dims = Tuple(mean_dims))
    return nothing
end

function _cross_correlation_averaged!(accumulator::AbstractVector{<:AbstractFloat}, a::AbstractVector{<:AbstractFloat}, b::AbstractVector{<:AbstractFloat})
    la = length(a)
    lb = length(b)
    l = min(la, lb)
    ls = typeof(a)(1:l)
    a_aug = fill_zeros(a,  1, lb-1, lb-1)
    b_aug = fill_zeros(b,  1,   0, la-1)
    a_pad = fill_ones(b, 1,   0, la-1)
    b_pad = fill_ones(a, 1, lb-1, lb-1)
    μa = similar(a_aug)
    μb = similar(b_aug)
    σa = similar(a_aug)
    σb = similar(b_aug)
    _accumulate!(μa, a_aug, a_pad)
    _accumulate!(μb, b_pad, b_aug)
    _accumulate!(σa, a_aug.^2, a_pad)
    _accumulate!(σb, b_aug.^2, b_pad)
    Accumulator = similar(a_aug)
    accumulate!(Accumulator, a_aug, b_aug)
    @inbounds σa[1:l] .-= abs2.(μa[1:l]) ./ls
    @inbounds σb[1:l] .-= abs2.(μb[1:l]) ./ls
    @inbounds accumulator .= (Accumulator[1:l] .- μa[1:l] .* μb[1:l] ./ ls)./sqrt(σa[1:l] .* σb[1:l])
    return nothing
end

# Stitch without shift: images MUST be the same size.

# Best case scenario for stitch with shift: images are the same size.
function _cross_correlation_averaged_!(accumulator::AbstractArray{<:AbstractFloat}, a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}})
    n = size(a)
    all_dims = 1:ndims(a)
    # @assert na[dims[1]] == nb[dims[1]]
    # @assert na[.!(in.(all_dims, [dims]))] == nb[.!(in.(all_dims, [dims]))]
    l = n[dims[1]]
    excluded_dims = all_dims[.!(in.(all_dims, [dims]))]
    mean_dims = vcat(excluded_dims, dims[2:end]...)
    a_aug = fill_zeros(a, dims[1], l-1, l-1)
    b_aug = fill_zeros(b, dims[1],   0, l-1)
    μa = fill_zeros(sum(a, dims = mean_dims), dims[1], l-1, l-1)
    μb = fill_zeros(sum(b, dims = mean_dims), dims[1], 0, l-1)
    σa = fill_zeros(sum(abs2, a, dims = mean_dims), dims[1], l-1, l-1)
    σb = fill_zeros(sum(abs2, b, dims = mean_dims), dims[1], 0, l-1)
    a_dim_min, a_dim_max, b_dim_min, b_dim_max, fft_dims_a, fft_dims_b = get_pad_indices_for_ones(n, n, dims[1])
    a_pad = fill_ones(μa, a_dim_min, a_dim_max)
    b_pad = fill_ones(μb, b_dim_min, b_dim_max)
    accumulate!(μa, a_pad, fft_dims_a)
    μb = accumulate(b_pad, μb, fft_dims_b)
    reverse!(μb, dims=dims[2:end])
    accumulate!(σa, a_pad, fft_dims_a)
    σb = accumulate(b_pad, σb, fft_dims_b)
    reverse!(σb, dims=dims[2:end])
    σab = accumulate(a_aug, b_aug, dims, excluded_dims)
    ls = typeof(μa)(calloc, [d == dims[1] ? l : 1 for d in all_dims]...)
    @inbounds ls[:] .= prod(n)*(1:l)/l
    ind_mean = [d == dims[1] ? (1:l) : 1:size(μa, d) for d in all_dims]
    ind_ab = [d == dims[1] ? (1:l) : d in dims ? (1:n[d]) : 1:1 for d in all_dims]
    @inbounds σa[ind_mean...] .-= abs2.(μa[ind_mean...]) ./ ls
    @inbounds σb[ind_mean...] .-= abs2.(μb[ind_mean...]) ./ ls
    @inbounds σab[ind_ab...] .-= (μa[ind_mean...] .* μb[ind_mean...]) ./ ls 
    @inbounds σab[ind_ab...] ./= sqrt.(σa[ind_mean...] .* σb[ind_mean...])
    accumulator .= dropdims(σab[ind_ab...], dims=Tuple(excluded_dims))
    return nothing
end

# This one is the worst case scenario, where the sizes of the input and output are different and you want a stitch with shift.
# The dimensions are MEANT to be different. The first dimension here is what I call the leading dimension. It is the dimension where the inputs MUST be the same size.
# I am not playing around here, this is an imposition for my sanity. No one is meant to come into contact with this anyway.
function _cross_correlation_averaged__!(accumulator::AbstractArray{<:AbstractFloat}, a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}})
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
    μa = fill_zeros(sum(a, dims = excluded_dims_a), dims[1], l-1, l-1)
    μb = fill_zeros(sum(b, dims = excluded_dims_b), dims[1], 0, l-1)
    σa = fill_zeros(sum(abs2, a, dims = excluded_dims_a), dims[1], l-1, l-1)
    σb = fill_zeros(sum(abs2, b, dims = excluded_dims_b), dims[1], 0, l-1)
    a_dim_min, a_dim_max, b_dim_min, b_dim_max, fft_dims_a, fft_dims_b = get_pad_indices_for_ones(na, nb, dims[1])
    a_pad = fill_ones(μa, a_dim_min, a_dim_max)
    b_pad = fill_ones(μb, b_dim_min, b_dim_max)
    accumulate!(μa, a_pad, fft_dims_a)
    μb = accumulate(b_pad, μb, fft_dims_b)
    reverse!(μb, dims=dims[2:end])
    accumulate!(σa, a_pad, fft_dims_a)
    σb = accumulate(b_pad, σb, fft_dims_b)
    reverse!(σb, dims=dims[2:end])
    fft_dims = all_dims[in.(all_dims, [fft_dims_a]) .|| in.(all_dims, [fft_dims_b])]
    σab = accumulate(a_aug, b_aug, fft_dims, excluded_dims)
    ls = typeof(μa)(calloc, [d == dims[1] ? l : 1 for d in all_dims]...)
    @inbounds ls[:] .= prod(min.(na, nb))*(1:l)/l
    ind_a = [d == dims[1] ? (1:l) : 1:size(μa, d) for d in all_dims]
    ind_b = [d == dims[1] ? (1:l) : 1:size(μb, d) for d in all_dims]
    ind_ab = [d == dims[1] ? (1:l) : d in dims ? (1:max(na[d], nb[d])) : 1:1 for d in all_dims]
    @inbounds σa[ind_a...] .-= abs2.(μa[ind_a...]) ./ ls
    @inbounds σb[ind_b...] .-= abs2.(μb[ind_b...]) ./ ls
    @inbounds σab[ind_ab...] .-= (μa[ind_a...] .* μb[ind_b...]) ./ ls 
    @inbounds σab[ind_ab...] ./= sqrt.(σa[ind_a...] .* σb[ind_b...])
    accumulator .= dropdims(σab[ind_ab...], dims=Tuple(excluded_dims))
    return nothing
end

function _cross_correlation_averaged!(accumulator::AbstractArray{<:AbstractFloat}, a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}})
    na = size(a)
    nb = size(b)
    @assert(na[dims[1]] == nb[dims[1]], "Inputs must have same size on the primary correlation dimension!")
    if na == nb
        _cross_correlation_averaged_!(accumulator, a, b, dims)
    else
        _cross_correlation_averaged__!(accumulator, a, b, dims)
    end
    return nothing
end

# Not in-place routines. These routines analyze the dimensions of the input and allocate the specific kind of output I want.

function _cross_correlation_averaged(a::AbstractArray{<:AbstractFloat, N}, b::AbstractArray{<:AbstractFloat, N}, dim::Integer) where {N}
    na = size(a)
    nb = size(b)
    l = min(na[dim], nb[dim])
    all_dims = 1:N
    @assert(na[all_dims .!= dim] == nb[all_dims .!= dim], "Cross-correlation on singleton dimensions requires arrays equally sized on the other dimensions!")
    accumulator = similar(a, l)
    _cross_correlation_averaged!(accumulator, a, b, dim)
    return accumulator
end

function _cross_correlation_averaged(a::AbstractArray{<:AbstractFloat, N}, b::AbstractArray{<:AbstractFloat, N}, dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}}) where {N}
    na = size(a)
    nb = size(b)
    l = min(na[dims[1]], nb[dims[1]])
    accumulator = similar(a, [d == dims[1] ? l : max(na[d], nb[d]) for d in dims]...)
    _cross_correlation_averaged!(accumulator, a, b, dims)
    return accumulator
end