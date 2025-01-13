## Cross correlation functions that accept array, images, vectors of arrays and vectors of images.
## The keyword arguments are used to choose between the various options: will the convolutions be calculated in RGB or grayscale? Will the computations be performed on the GPU (WARNING: CUDA ONLY) or on the CPU?  Etc.

phat_keys = [:phat, :Phat, :PHAT, :PhaseTransform, :phasetransform]
averaged_keys = [:averaged, :average, :avrg, :Averaged, :Average, :AVERAGED, :AVERAGE, :AVRG]
none_keys = [:none, :None, :NONE]

rgb_keys = [:rgb, :RGB, :RedGreenBlue, :REDGREENBLUE]
gray_keys = [:none, :None, :NONE, :Gray, :gray, :GRAY]

down_keys = [:down, :Down, :DOWN, :down_up, :Down_Up, :DOWN_UP]
up_keys = [:up, :Up, :UP, :UP_DOWN, :Up_Down, :up_down]
left_keys = [:left, :Left, :LEFT, :LEFT_RIGHT, :Left_Right, :left_right]
right_keys = [:right, :Right, :RIGHT, :RIGHT_LEFT, :Right_Left, :right_left]


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


function _generalized_cross_correlation_none(a::AbstractArray{<:AbstractFloat, N}, b::AbstractArray{<:AbstractFloat, N}, dim::Union{Integer, NTuple{<:Any, Integer}, Vector{<:Integer}}) where {N}
    @match N begin
        1 => _cross_correlation(a, b)
        2 => _cross_correlation(a, b, dim)
        3 => _cross_correlation(a,b,dim .+ 1)
        _ => error("Dimension not supported for correlation of images!")
    end
end

function _generalized_cross_correlation_averaged(a::AbstractArray{<:AbstractFloat, N}, b::AbstractArray{<:AbstractFloat, N}, dim::Union{Integer, NTuple{<:Any, Integer}, Vector{<:Integer}}) where {N}
    @match N begin
        1 => _cross_correlation_averaged(a, b)
        2 => _cross_correlation_averaged(a, b, dim)
        3 => _cross_correlation_averaged(a,b,dim .+ 1)
        _ => error("Dimension not supported for correlation of images!")
    end
end

function _generalized_cross_correlation_PHAT(a::AbstractArray{<:AbstractFloat, N}, b::AbstractArray{<:AbstractFloat, N}, dim::Union{Integer, NTuple{<:Any, Integer}, Vector{<:Integer}}) where {N}
    @match N begin
        1 => _cross_correlation_PHAT(a, b)
        2 => _cross_correlation_PHAT(a, b, dim)
        3 => _cross_correlation_PHAT(a,b,dim .+ 1)
        _ => error("Dimension not supported for correlation of images!")
    end
end

function _generalized_cross_correlation(a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}, dim::Union{Integer, NTuple{<:Any, Integer}, Vector{<:Integer}}; type::Symbol=:none)
    if type in none_keys 
        return _generalized_cross_correlation_none(a, b, dim)
    elseif type in averaged_keys
        return _generalized_cross_correlation_averaged(a,b,dim)
    elseif type in phat_keys
        return _generalized_cross_correlation_PHAT(a,b,dim)
    else
        error("Invalid type for correlation!\n For no special correlation algorithm, try ", none_keys, ".\n For moving average correlation, try ", averaged_keys, ".\n And for PHAT correlations, try ", phat_keys, ".")
    end
end

function generalized_cross_correlation(a::AbstractMatrix{<:RGB{T}}, b::AbstractMatrix{<:RGB{T}}, dim::Union{Integer, NTuple{<:Any, Integer}, Vector{<:Integer}}; type::Symbol=:none, gpu = false, rgb::Symbol = :Gray) where {T<:Real}
    na = size(a)
    nb = size(b)
    inds_a, inds_b = _get_indexes(na, nb, dim)
    if rgb in rgb_keys && gpu
        A = GPUArray(float(channelview(a[inds_a...])))
        B = GPUArray(float(channelview(b[inds_b...])))
    elseif rgb in rgb_keys && !gpu
        A = float(channelview(a[inds_a...]))
        B = float(channelview(b[inds_b...]))
    elseif rgb in gray_keys && gpu
        A = GPUMatrix(float(T).(Gray.(a[inds_a...])))
        B = GPUMatrix(float(T).(Gray.(b[inds_b...])))
    elseif rgb in gray_keys && !gpu
        A = float(T).(Gray.(a[inds_a...]))
        B = float(T).(Gray.(b[inds_b...]))
    else
        error("Invalid option for color type!\n Use one of ", rgb_keys, "\n or use ", gray_keys, "for grayscale processing!")
    end
    return _generalized_cross_correlation(A, B, dim, type=type)
end

function generalized_cross_correlation(a::AbstractMatrix{<:RGB{T}}, b::AbstractMatrix{<:RGB{T}}; dim::Symbol, type::Symbol=:none, gpu = false, rgb::Symbol = :Gray) where {T<:Real}
    if dim in down_keys
        return generalized_cross_correlation(a, b, 1, type = type, gpu = gpu, rgb = rgb)
    elseif dim in up_keys
        return generalized_cross_correlation(b, a, 1, type = type, gpu = gpu, rgb = rgb)
    elseif dim in right_left
        return generalized_cross_correlation(a, b, 2, type = type, gpu = gpu, rgb = rgb)
    elseif dim in left_right
        return generalized_cross_correlation(b, a, 2, type = type, gpu = gpu, rgb = rgb)
    else
        error("Invalid code for dimensional input!")
    end
end

function generalized_cross_correlation_shift(a::AbstractMatrix{<:RGB{T}}, b::AbstractMatrix{<:RGB{T}}, dim::Integer; type::Symbol=:none, gpu = false, rgb::Symbol = :Gray) where {T<:Real}
    na = size(a)
    nb = size(b)
    dims = dim == 1 ? (1,2) : dim == 2 ? (2,1) : error("Invalid leading dimension for image correlation")
    inds_a, inds_b = _get_indexes(na, nb, dims)
    if rgb in rgb_keys && gpu
        A = GPUArray(float(channelview(a[inds_a...])))
        B = GPUArray(float(channelview(b[inds_b...])))
    elseif rgb in rgb_keys && !gpu
        A = float(channelview(a[inds_a...]))
        B = float(channelview(b[inds_b...]))
    elseif rgb in gray_keys && gpu
        A = GPUMatrix(float(T).(Gray.(a[inds_a...])))
        B = GPUMatrix(float(T).(Gray.(b[inds_b...])))
    elseif rgb in gray_keys && !gpu
        A = float(T).(Gray.(a[inds_a...]))
        B = float(T).(Gray.(b[inds_b...]))
    else
        error("Invalid option for color type!\n Use one of ", rgb_keys, "\n or use ", gray_keys, "for grayscale processing!")
    end
    return _generalized_cross_correlation(A, B, dims, type=type)
end

function generalized_cross_correlation_shift(a::AbstractMatrix{<:RGB{T}}, b::AbstractMatrix{<:RGB{T}}, dim::Union{Integer, NTuple{<:Any, Integer}, Vector{<:Integer}}; type::Symbol=:none, gpu = false, rgb::Symbol = :Gray) where {T<:Real}
    na = size(a)
    nb = size(b)
    inds_a, inds_b = _get_indexes(na, nb, dim)
    if rgb in rgb_keys && gpu
        A = GPUArray(float(channelview(a[inds_a...])))
        B = GPUArray(float(channelview(b[inds_b...])))
    elseif rgb in rgb_keys && !gpu
        A = float(channelview(a[inds_a...]))
        B = float(channelview(b[inds_b...]))
    elseif rgb in gray_keys && gpu
        A = GPUMatrix(float(T).(Gray.(a[inds_a...])))
        B = GPUMatrix(float(T).(Gray.(b[inds_b...])))
    elseif rgb in gray_keys && !gpu
        A = float(T).(Gray.(a[inds_a...]))
        B = float(T).(Gray.(b[inds_b...]))
    else
        error("Invalid option for color type!\n Use one of ", rgb_keys, "\n or use ", gray_keys, "for grayscale processing!")
    end
    return _generalized_cross_correlation(A, B, dim, type=type)
end

function generalized_cross_correlation_shift(a::AbstractMatrix{<:RGB{T}}, b::AbstractMatrix{<:RGB{T}}; dim::Symbol, type::Symbol=:none, gpu = false, rgb::Symbol = :Gray) where {T<:Real}
    if dim in down_keys
        return generalized_cross_correlation(a, b, (1,2), type = type, gpu = gpu, rgb = rgb)
    elseif dim in up_keys
        return generalized_cross_correlation(b, a, (1,2), type = type, gpu = gpu, rgb = rgb)
    elseif dim in right_left
        return generalized_cross_correlation(a, b, (2,1), type = type, gpu = gpu, rgb = rgb)
    elseif dim in left_right
        return generalized_cross_correlation(b, a, (2,1), type = type, gpu = gpu, rgb = rgb)
    else
        error("Invalid code for dimensional input!")
    end
end
