# Transition functions for the application of feathering. Usually, feathering uses only linear transitions, but I've often found that the linear stuff is not sufficient.

# The first function we explore here is the standard feathering, but generalized. By introducing a transition variable γ that changes the transition to either favor the top (γ >0) or to favor the bottom (γ<0) with γ=0 being a neutral midpoint, the linear transition.
# This function is continuous and always maps 0 to 0 as well as 1 to 1 for any positive γ.
function feathers(t::T; γ::Real=0.0)::float(T) where {T<:Real}
    if iszero(γ)
        return t
    else
        return (exp(γ*t)-1)/(exp(γ)-1)
    end
end

function feathers(t::Real, gamma::Real)
    return feathers(t, γ = gamma)
end

# The smoothest mixer is a mixer that has the property of preserving all derivatives at the stitch point in a 1D stitch. This function is infinitely differentiable for any positive α or β and guarantees a smooth transition between images, but sometimes it can be too smooth and make edges blurry. This weight maps any value <0 to 0 and >β to 1. Greater β leads to a lower stitch line, greater α leads to a faster transition. It is best to think as 0<β<1.
function smoothest_mixer(t::T; α::Real = 1.0, β::Real = 1.0)::float(T) where {T<:Real}
    if t ≤ 0.0
        return 0.0
    elseif 0.0 < t < β
        ϕt = exp(-1/(t)^α)
        ϕβt = exp(-1/(β-t)^α)
        return ϕt/(ϕt + ϕβt)
    else
        return 1.0
    end
end

function smoothest_mixer(t::Real, alpha::Real, beta::Real)
    return smoothest_mixer(t, α = alpha, β = beta)
end

# I've found that the most convenient way of doing feathering is to just use both at the same time: use the smoothest mixer to smooth over the transition 

function smoothest_feathers(t::Real;α::Real = 1.0, β::Real = 1.0, γ::Real=0.0)
    return smoothest_mixer(feathers(t, γ=γ), α = α, β=β)
end

function smoothest_feathers(t::Real,alpha::Real, beta::Real, gamma::Real)
    return smoothest_mixer(feathers(t, γ=gamma), α = alpha, β=beta)
end

T = Float32

Gray.([zeros(T, 1080, 1960);feathers.(repeat(reshape(0:1000, :, 1), 1, 1960)/1000, γ = 2.0);ones(T, 1080, 1960)])
Gray.([zeros(T, 1080, 1960);smoothest_mixer.(repeat(reshape(0:1000, :, 1), 1, 1960)/1000, 1.8, 0.9);ones(T, 1080, 1960)])
Gray.([zeros(T, 1080, 1960);smoothest_feathers.(repeat(reshape(0:1000, :, 1), 1, 1960)/1000, 0.8, 0.9, 2.0);ones(T, 1080, 1960)])