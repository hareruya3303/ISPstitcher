function _accumulate!(accumulator::AbstractVector{<:AbstractFloat}, a::AbstractVector{<:AbstractFloat}, b::AbstractVector{<:AbstractFloat})
    A = rfft(a)
    B = rfft(b)
    @inbounds accumulator .= irfft(A .* conj(B), length(a))
    return nothing
end

function accumulate!(a::AbstractVector{<:AbstractFloat}, b::AbstractVector{<:AbstractFloat})
    A = rfft(a)
    B = rfft(b)
    @inbounds a .= irfft(A .* conj(B), length(a))
    return nothing
end

function _accumulate_PHAT!(accumulator::AbstractVector{<:AbstractFloat}, a::AbstractVector{<:AbstractFloat}, b::AbstractVector{<:AbstractFloat})
    A = rfft(a)
    B = rfft(b)
    @inbounds accumulator .= irfft(sign.(A .* conj(B)), length(a))
    return nothing
end

function accumulate_PHAT!(a::AbstractVector{<:AbstractFloat}, b::AbstractVector{<:AbstractFloat})
    A = rfft(a)
    B = rfft(b)
    @inbounds a .= irfft(sign.(A .* conj(B)), length(a))
    return nothing
end

function _accumulate!(accumulator::AbstractArray{<:AbstractFloat}, a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, accumulation_dim::Integer)
    dim = ndims(a)
    all_dims = 1:dim
    permuted_dims = vcat(all_dims[all_dims .!= accumulation_dim]..., accumulation_dim)
    unpermuted_dims = unpermute_indices(permuted_dims)
    planner = plan_rfft(permutedims(a, permuted_dims), dim)
    A = planner*permutedims(a, permuted_dims)
    B = planner*permutedims(b, permuted_dims)
    @inbounds accumulator .= permutedims(planner \(A .* conj(B)), unpermuted_dims)
    return nothing
end

function _accumulate!(accumulator::AbstractArray{<:AbstractFloat}, a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, accumulation_dim::Integer, reduction_dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}})
    # accumulator: multidimensional array where the resulting calculations are stored.
    # accumulation_dim: dimension where the output will be accumulated.
    # reduction_dim: dimensions where the output will be reduced and added, to reduce computational load on the system.
    dim = ndims(a)
    all_dims = 1:dim
    l_r = length(reduction_dims)
    ignored_dims = all_dims[.!(in.(all_dims, [vcat(reduction_dims..., accumulation_dim)]))]
    permuted_dims = vcat(ignored_dims..., reduction_dims..., accumulation_dim)
    unpermuted_dims = unpermute_indices(permuted_dims)
    A = rfft(permutedims(a, permuted_dims), dim)
    B = rfft(permutedims(b, permuted_dims), dim)
    @inbounds accumulator .= permutedims(irfft(sum(A .* conj(B), dims = (dim-l_r):(dim-1)), size(a, accumulation_dim), dim), unpermuted_dims)
    return nothing
end

function _accumulate_PHAT!(accumulator::AbstractArray{<:AbstractFloat}, a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, accumulation_dim::Integer)
    dim = ndims(a)
    all_dims = 1:dim
    permuted_dims = vcat(all_dims[all_dims .!= accumulation_dim]..., accumulation_dim)
    unpermuted_dims = unpermute_indices(permuted_dims)
    planner = plan_rfft(permutedims(a, permuted_dims), dim)
    A = planner*permutedims(a, permuted_dims)
    B = planner*permutedims(b, permuted_dims)
    @inbounds accumulator .= permutedims(planner \sign.(A .* conj(B)), unpermuted_dims)
    return nothing
end

function _accumulate_PHAT!(accumulator::AbstractArray{<:AbstractFloat}, a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, accumulation_dim::Integer, reduction_dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}})
    dim = ndims(a)
    all_dims = 1:dim
    fft_dims = vcat(accumulation_dim, reduction_dims...)
    l = length(fft_dims)
    ignored_dims = all_dims[.!(in.(all_dims, [fft_dims]))]
    permuted_dims = vcat(ignored_dims..., fft_dims...)
    unpermuted_dims = unpermute_indices(permuted_dims)
    A = rfft(permutedims(a, permuted_dims), (dim-l+1):dim)
    B = rfft(permutedims(b, permuted_dims), (dim-l+1):dim)
    @inbounds accumulator .= permutedims(irfft(mean(sign.(A .*conj(B)), dims = (dim-l+2):dim), size(a, accumulation_dim), dim-l+1), unpermuted_dims)
    return nothing
end

function accumulate!(a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, accumulation_dim::Integer)
    dim = ndims(a)
    all_dims = 1:dim
    permuted_dims = vcat(all_dims[all_dims .!= accumulation_dim]..., accumulation_dim)
    unpermuted_dims = unpermute_indices(permuted_dims)
    planner = plan_rfft(permutedims(a, permuted_dims), dim)
    A = planner*permutedims(a, permuted_dims)
    B = planner*permutedims(b, permuted_dims)
    @inbounds a .= permutedims(planner \(A .* conj(B)), unpermuted_dims)
    return nothing
end

function accumulate_PHAT!(a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, accumulation_dim::Integer)
    dim = ndims(a)
    all_dims = 1:dim
    permuted_dims = vcat(all_dims[all_dims .!= accumulation_dim]..., accumulation_dim)
    unpermuted_dims = unpermute_indices(permuted_dims)
    planner = plan_rfft(permutedims(a, permuted_dims), dim)
    A = planner*permutedims(a, permuted_dims)
    B = planner*permutedims(b, permuted_dims)
    @inbounds a .= permutedims(planner \sign.(A .* conj(B)), unpermuted_dims)
    return nothing
end

function _accumulate!(accumulator::AbstractArray{<:AbstractFloat}, a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, accumulation_dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}})
    dim = ndims(a)
    ignored_dims = filter(x->~(x in accumulation_dims), 1:dim)
    fft_dims = Tuple((dim-length(accumulation_dims)+1):dim)
    permuted_dims = vcat(ignored_dims..., accumulation_dims...)
    unpermuted_dims = unpermute_indices(permuted_dims)
    planner = plan_rfft(permutedims(a, permuted_dims), fft_dims)
    A = planner*permutedims(a, permuted_dims)
    B = planner*permutedims(b, permuted_dims)
    @inbounds accumulator .= permutedims(planner \(A .* conj(B)), unpermuted_dims)
    return nothing
end

function _accumulate!(accumulator::AbstractArray{<:AbstractFloat}, a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, accumulation_dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}}, reduction_dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}})
    dim = ndims(a)
    ignored_dims = filter(x->!((x in accumulation_dims)||(x in reduction_dims)), 1:dim)
    l_a = length(accumulation_dims)
    l_r = length(reduction_dims)
    fft_dims = (dim-l_a+1):dim
    reduct_dims = (dim - l_a - l_r + 1):(dim - l_a)
    permuted_dims = (ignored_dims..., reduction_dims..., accumulation_dims...)
    unpermuted_dims = unpermute_indices(permuted_dims)
    A = rfft(permutedims(a, permuted_dims), fft_dims)
    B = rfft(permutedims(b, permuted_dims), fft_dims)
    @inbounds accumulator .= permutedims(irfft(sum(A .* conj.(B), dims = reduct_dims), size(a, accumulation_dims[1]), dim-l_a+1:dim), unpermuted_dims)
    return nothing
end

function _accumulate_PHAT!(accumulator::AbstractArray{<:AbstractFloat}, a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, accumulation_dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}})
    dim = ndims(a)
    ignored_dims = filter(x->~(x in accumulation_dims), 1:dim)
    fft_dims = Tuple((dim-length(accumulation_dims)+1):dim)
    permuted_dims = (ignored_dims..., accumulation_dims...)
    unpermuted_dims = unpermute_indices(permuted_dims)
    planner = plan_rfft(permutedims(a, permuted_dims), fft_dims)
    A = planner*permutedims(a, permuted_dims)
    B = planner*permutedims(b, permuted_dims)
    @inbounds accumulator .= permutedims(planner \sign.(A .* conj(B)), unpermuted_dims)
    return nothing
end

function _accumulate_PHAT!(accumulator::AbstractArray{<:AbstractFloat}, a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, accumulation_dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}}, reduction_dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}})
    dim = ndims(a)
    l_a = length(accumulation_dims)
    l_r = length(reduction_dims)
    ignored_dims = filter(x->!((x in accumulation_dims)||(x in reduction_dims)), 1:dim)
    fft_dims = Tuple((dim-l_r-l_a+1):dim)
    permuted_dims = (ignored_dims..., accumulation_dims..., reduction_dims...)
    unpermuted_dims = unpermute_indices(permuted_dims)
    A = rfft(permutedims(a, permuted_dims), fft_dims)
    B = rfft(permutedims(b, permuted_dims), fft_dims)
    @inbounds accumulator .= permutedims(irfft(mean(sign.(A .* conj(B)), dims=(dim-l_r+1):dim), size(a, accumulation_dims[1]), (dim-l_a-l_r+1):(dim-l_r)), unpermuted_dims)
    return nothing
end

function accumulate!(a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, accumulation_dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}})
    dim = ndims(a)
    ignored_dims = filter(x->~(x in accumulation_dims), 1:dim)
    fft_dims = Tuple((dim-length(accumulation_dims)+1):dim)
    permuted_dims = (ignored_dims..., accumulation_dims..., )
    unpermuted_dims = unpermute_indices(permuted_dims)
    planner = plan_rfft(permutedims(a, permuted_dims), fft_dims)
    A = planner*permutedims(a, permuted_dims)
    B = planner*permutedims(b, permuted_dims)
    @inbounds a .= permutedims(planner \(A .* conj(B)), unpermuted_dims)
    return nothing
end

function _accumulate_PHAT!(a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, accumulation_dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}})
    dim = ndims(a)
    ignored_dims = filter(x->~(x in accumulation_dims), 1:dim)
    fft_dims = Tuple((dim-length(accumulation_dims)+1):dim)
    permuted_dims = (ignored_dims..., accumulation_dims..., )
    unpermuted_dims = unpermute_indices(permuted_dims)
    planner = plan_rfft(permutedims(a, permuted_dims), fft_dims)
    A = planner*permutedims(a, permuted_dims)
    B = planner*permutedims(b, permuted_dims)
    @inbounds a .= permutedims(planner \sign.(A .* conj(B)), unpermuted_dims)
    return nothing
end

function accumulate(a::AbstractVector{<:AbstractFloat}, b::AbstractVector{<:AbstractFloat})
    accumulator = similar(a)
    _accumulate!(accumulator, a, b)
    return accumulator
end

function accumulate_PHAT(a::AbstractVector{<:AbstractFloat}, b::AbstractVector{<:AbstractFloat})
    accumulator = similar(a)
    _accumulate!(accumulator, a, b)
    return accumulator
end

function accumulate(a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, accumulation_dim::Integer, reduction_dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}} = Int[])
    if isempty(reduction_dims)
        accumulator = similar(a)
        _accumulate!(accumulator, a, b, accumulation_dim)
    else
        accumulator = similar(a, [d in reduction_dims ? 1 : size(a, d) for d in 1:ndims(a)]...)
        _accumulate!(accumulator, a, b, accumulation_dim, reduction_dims)
    end
    return accumulator
end

function accumulate_PHAT(a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, accumulation_dim::Integer, reduction_dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}} = Int[])
    if isempty(reduction_dims)
        accumulator = similar(a)
        _accumulate_PHAT!(accumulator, a, b, accumulation_dim)
    else
        accumulator = similar(a, [d in reduction_dims ? 1 : size(a, d) for d in 1:ndims(a)]...)
        _accumulate_PHAT!(accumulator, a, b, accumulation_dim, reduction_dims)
    end
    return accumulator
end

function accumulate(a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, accumulation_dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}}, reduction_dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}} = Int[])
    if isempty(reduction_dims)
        accumulator = similar(a)
        _accumulate!(accumulator, a, b, accumulation_dims)
    else
        accumulator = similar(a, [d in reduction_dims ? 1 : size(a, d) for d in 1:ndims(a)]...)
        _accumulate!(accumulator, a, b, accumulation_dims, reduction_dims)
    end
    return accumulator
end

function accumulate_PHAT(a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, accumulation_dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}}, reduction_dims::Union{NTuple{<:Any, Integer}, Vector{<:Integer}} = Int[])
    if isempty(reduction_dims)
        accumulator = similar(a)
        _accumulate_PHAT!(accumulator, a, b, accumulation_dims)
    else
        accumulator = similar(a, [d in reduction_dims ? 1 : size(a, d) for d in 1:ndims(a)]...)
        _accumulate_PHAT!(accumulator, a, b, accumulation_dims, reduction_dims)
    end
    return accumulator
end

