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
