
function forwardrecurrence!(v::AbstractVector, b::AbstractVector, a::AbstractVector, c::AbstractVector, x, shift=0)
    isempty(v) && return v
    p0 = one(x) # assume OPs are normalized to one for now
    p1 = (x-a[1])/c[1]
    @inbounds for n = 1:shift
        p1,p0 = muladd(x-a[n-1],v[n-1],-b[n-1]*v[n-2])/c[n-1],p1
    end
    v[1] = p0
    length(v) == 1 && return v
    v[2] = p1
    @inbounds for n = 3:length(v)
        p1,p0 = muladd(x-a[n-1],v[n-1],-b[n-1]*v[n-2])/c[n-1],p1
        v[n] = p1
    end
    v
end

function forwardrecurrence!(v::AbstractVector, b::AbstractVector, ::Zeros{<:Any,1}, c::AbstractVector, x, shift=0)
    isempty(v) && return v
    p0 = one(x) # assume OPs are normalized to one for now
    p1 = x/c[1]
    @inbounds for n = 1:shift
        p1,p0 = muladd(x,p1,-b[n-1]*p0)/c[n-1],p1
    end
    v[1] = p0
    length(v) == 1 && return v
    v[2] = p1
    @inbounds for n = 3:length(v)
        p1,p0 = muladd(x,p1,-b[n-1]*p0)/c[n-1],p1
        v[n] = p1
    end
    v
end

# special case for Chebyshev
function forwardrecurrence!(v::AbstractVector, b_v::AbstractFill, ::Zeros{<:Any,1}, c::Vcat{<:Any,1,<:Tuple{<:Number,<:AbstractFill}}, x, shift=0)
    isempty(v) && return v
    c0,c∞_v = c.args
    b = getindex_value(b_v)
    c∞ = getindex_value(c∞_v)
    mbc  = -b/c∞
    xc = x/c∞
    p0 = one(x) # assume OPs are normalized to one for now
    p1 = x/c0
    for n = 1:shift
        p1,p0 = muladd(xc,p1,mbc*p0),p1
    end
    v[1] = p0
    length(v) == 1 && return v
    v[2] = p1
    @inbounds for n = 3:length(v)
        p1,p0 = muladd(xc,p1,mbc*p0),p1
        v[n] = p1
    end
    v
end