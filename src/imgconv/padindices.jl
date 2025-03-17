function _determine_padding_conv(na::NTuple{N, I}, nb::NTuple{N, I}, pad_dim::Integer) where {N, I<:Integer}
    pad_dims_a = I[]
    pad_dims_b = I[]
    padval_a = I[]
    padval_b = I[]
    Δpad_a = I[]
    excluded_dims = I[]
    for dim in eachindex(na)
        if dim == pad_dim
            push!(pad_dims_a, dim)
            push!(pad_dims_b, dim)
            push!(Δpad_a, nb[dim]-1)
            push!(padval_a, nb[dim]-1)
            push!(padval_b, na[dim]-1)
        elseif na[dim]>nb[dim]
            push!(pad_dims_b, dim)
            push!(padval_b, na[dim]-nb[dim])
        elseif nb[dim]>na[dim]
            push!(pad_dims_a, dim)
            push!(Δpad_a, 0)
            push!(padval_a, nb[dim]-na[dim])
        else
            push!(excluded_dims, dim)
        end
    end
    Δpad_b = zero.(pad_dims_b)
    return pad_dims_a, pad_dims_b, Δpad_a, Δpad_b, padval_a, padval_b, excluded_dims
end

function get_pad_indices_for_ones(na::NTuple{N, I}, nb::NTuple{N, I}, pad_dim::Integer) where {N, I<:Integer}
    ndim = length(na)
    μa_dim_min = Vector{I}(calloc, ndim)
    μa_dim_max = copy(μa_dim_min)
    μb_dim_min = copy(μa_dim_min)
    μb_dim_max = copy(μa_dim_min)
    fft_dims_a = I[]
    fft_dims_b = I[]
    for dim in eachindex(na)
        if dim == pad_dim
            μa_dim_min[dim] = 1
            μa_dim_max[dim] = nb[dim]
            μb_dim_min[dim] = nb[dim]
            μb_dim_max[dim] = na[dim]+nb[dim]-1   
            push!(fft_dims_a, dim)
            push!(fft_dims_b, dim)         
        elseif na[dim]>nb[dim]
            μa_dim_min[dim] = 1
            μa_dim_max[dim] = nb[dim]
            μb_dim_min[dim] = 1
            μb_dim_max[dim] = 1
            push!(fft_dims_a, dim)
        elseif nb[dim]>na[dim]
            μa_dim_min[dim] = 1
            μa_dim_max[dim] = 1
            μb_dim_min[dim] = 1
            μb_dim_max[dim] = na[dim]
            push!(fft_dims_b, dim)
        else
            μa_dim_min[dim] = 1
            μa_dim_max[dim] = 1
            μb_dim_min[dim] = 1
            μb_dim_max[dim] = 1
        end
    end
    return μa_dim_min, μa_dim_max, μb_dim_min, μb_dim_max, fft_dims_a, fft_dims_b
end
    

function unpermute_indices(permuted::Union{NTuple{N, <:Integer}, Vector{<:Integer}}) where {N}
    unpermuted = vcat(permuted...)
    for t in eachindex(permuted)
        unpermuted[permuted[t]] = t
    end
    return unpermuted
end

function _get_indexes(na::Tuple{Integer, Integer}, nb::Tuple{Integer, Integer}, dim::Integer)
    if dim ==1
        y = min(na[1], nb[1])
        return [1:y, :], [nb[1]-y+1:nb[1], :]
    elseif dim ==2
        x = min(na[2], nb[2])
        return [:, 1:x], [:, (nb[2]-x+1):nb[2]]
    else
        error("Unsupported dimension! Images are considered to be 2D only!")
    end
end

function _get_indexes(na::Tuple{Integer, Integer}, nb::Tuple{Integer, Integer}, dim::Union{NTuple{N, Integer}, Vector{<:Integer}}) where {N}
    if dim == (1,2)
        y = min(na[1], nb[1])
        return [1:y, :], [(nb[1]-y+1):nb[1], :]
    elseif dim == (2,1)
        x = min(na[2], nb[2])
        return [:, 1:x], [:, (nb[2]-x+1):nb[2]]
    else
        error("Unsupported dimension! Images are considered to be 2D only!")
    end
end

function __make_indexes(na, nb, dims)
    acc_dims = Vector{Int}(calloc, length(dims))
    reord_dims = Vector{Vector{Int}}(undef, length(dims))
    index_dims = Vector{UnitRange{Int}}(undef, length(dims))
    sorted_dims = sort(collect(dims))
    for t in sorted_dims
        d = sorted_dims[t]
        if d == dims[1]
            l = min(na[d], nb[d])
            acc_dims[t] = l
            reord_dims[t] = 1:l
            index_dims[t] = 1:l
        else
            L = max(na[d], nb[d])
            acc_dims[t] = L
            reord_dims[t] = vcat((div(L, 2)+1):L, 1:div(L, 2))
            index_dims[t] = (div(L, 2) - L):(div(L, 2) - 1)
        end
    end
    return acc_dims, reord_dims, index_dims
end

    