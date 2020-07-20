
##
# For Chebyshev T. Note the shift in indexing is fine due to the AbstractFill
##
Base.@propagate_inbounds _forwardrecurrence_next(n, A::Vcat{<:Any,1,<:Tuple{<:Number,<:AbstractFill}}, B::Zeros, C::Ones, x, p0, p1) = 
    _forwardrecurrence_next(n, A.args[2], B, C, x, p0, p1)

Base.@propagate_inbounds _clenshaw_next(n, A::Vcat{<:Any,1,<:Tuple{<:Number,<:AbstractFill}}, B::Zeros, C::Ones, x, c, bn1, bn2) = 
    _clenshaw_next(n, A.args[2], B, C, x, c, bn1, bn2)

function initiateforwardrecurrence(N, A, B, C, x)
    T = promote_type(eltype(A), eltype(B), eltype(C), typeof(x))
    p0 = one(T)
    p1 = convert(T, A[1]x + B[1])
    @inbounds for n = 2:N
        p1,p0 = _forwardrecurrence_next(n, A, B, C, x, p0, p1),p1
    end
    p0,p1
end

getindex(P::OrthogonalPolynomial{T}, x::Number, n::OneTo) where T =
    copyto!(Vector{T}(undef,length(n)), view(P, x, n))

getindex(P::OrthogonalPolynomial{T}, x::AbstractVector, n::AbstractUnitRange{Int}) where T =
    copyto!(Matrix{T}(undef,length(x),length(n)), view(P, x, n))

function copyto!(dest::AbstractArray, V::SubArray{<:Any,1,<:OrthogonalPolynomial,<:Tuple{<:Number,<:OneTo}})
    P = parent(V)
    x,n = parentindices(V)
    A,B,C = recurrencecoefficients(P)
    forwardrecurrence!(dest, A, B, C, x)
end

function copyto!(dest::AbstractArray, V::SubArray{<:Any,2,<:OrthogonalPolynomial,<:Tuple{<:AbstractVector,<:UnitRange}})
    checkbounds(dest, axes(V)...)
    P = parent(V)
    xr,jr = parentindices(V)
    A,B,C = recurrencecoefficients(P)
    shift = first(jr)
    Ã,B̃,C̃ = A[shift:∞],B[shift:∞],C[shift:∞]
    for (k,x) = enumerate(xr)
        p0, p1 = initiateforwardrecurrence(shift, A, B, C, x)
        _forwardrecurrence!(view(dest,k,:), Ã, B̃, C̃, x, p0, p1)
    end
    dest
end

function copyto!(dest::AbstractArray, V::SubArray{<:Any,1,<:OrthogonalPolynomial,<:Tuple{<:Number,<:UnitRange}})
    checkbounds(dest, axes(V)...)
    P = parent(V)
    x,jr = parentindices(V)
    A,B,C = recurrencecoefficients(P)
    shift = first(jr)
    Ã,B̃,C̃ = A[shift:∞],B[shift:∞],C[shift:∞]
    p0, p1 = initiateforwardrecurrence(shift, A, B, C, x)
    _forwardrecurrence!(dest, Ã, B̃, C̃, x, p0, p1)
    dest
end

getindex(P::OrthogonalPolynomial, x::Number, n::UnitRange) = layout_getindex(P, x, n)
getindex(P::OrthogonalPolynomial, x::AbstractVector, n::UnitRange) = layout_getindex(P, x, n)

getindex(P::OrthogonalPolynomial, x::Number, n::AbstractVector{<:Integer}) =
    P[x,OneTo(maximum(n))][n]

getindex(P::OrthogonalPolynomial, x::AbstractVector, n::AbstractVector{<:Integer}) =
    P[x,OneTo(maximum(n))][:,n]

getindex(P::OrthogonalPolynomial, x::Number, n::Number) = P[x,OneTo(n)][end]



###
# Clenshaw
###

function getindex(f::Expansion{<:Any,<:OrthogonalPolynomial}, x::Number)
    P,c = arguments(f)
    clenshaw(paddeddata(c), recurrencecoefficients(P)..., x)
end

getindex(f::Expansion{T,<:OrthogonalPolynomial}, x::AbstractVector{<:Number}) where T = 
    copyto!(Vector{T}(undef, length(x)), view(f, x))

function copyto!(dest::AbstractVector{T}, v::SubArray{<:Any,1,<:Expansion{<:Any,<:OrthogonalPolynomial}, <:Tuple{AbstractVector{<:Number}}}) where T
    f = parent(v)
    (x,) = parentindices(v)
    P,c = arguments(f)
    clenshaw!(paddeddata(c), recurrencecoefficients(P)..., x, Ones{T}(length(x)), dest)
end

###
# Operator clenshaw
###


Base.@propagate_inbounds function _clenshaw_next!(n, A::AbstractFill, ::Zeros, C::Ones, x::AbstractMatrix, c, bn1::AbstractMatrix{T}, bn2::AbstractMatrix{T}) where T
    muladd!(getindex_value(A), x, bn1, -one(T), bn2)
    view(bn2,band(0)) .+= c[n]
    bn2
end

Base.@propagate_inbounds function _clenshaw_next!(n, A::AbstractVector, ::Zeros, C::AbstractVector, x::AbstractMatrix, c, bn1::AbstractMatrix{T}, bn2::AbstractMatrix{T}) where T
    muladd!(A[n], x, bn1, -C[n+1], bn2)
    view(bn2,band(0)) .+= c[n]
    bn2
end

# Operator * f Clenshaw
Base.@propagate_inbounds function _clenshaw_next!(n, A::AbstractFill, ::Zeros, C::Ones, x::AbstractMatrix, c, f::AbstractVector, bn1::AbstractVector{T}, bn2::AbstractVector{T}) where T
    muladd!(getindex_value(A), x, bn1, -one(T), bn2)
    bn2 .+= c[n] .* f
    bn2
end

# allow special casing first arg, for ChebyshevT in OrthogonalPolynomialsQuasi
Base.@propagate_inbounds function _clenshaw_first!(A, ::Zeros, C, X, c, bn1, bn2) 
    muladd!(A[1], X, bn1, -C[2], bn2)
    view(bn2,band(0)) .+= c[1]
    bn2
end

Base.@propagate_inbounds function _clenshaw_first!(A, ::Zeros, C, X, c, f::AbstractVector, bn1, bn2) 
    muladd!(A[1], X, bn1, -C[2], bn2)
    bn2 .+= c[1] .* f
    bn2
end

_clenshaw_op(::AbstractBandedLayout, Z, N) = BandedMatrix(Z, (N-1,N-1))

function clenshaw(c::AbstractVector, A::AbstractVector, B::AbstractVector, C::AbstractVector, X::AbstractMatrix)
    N = length(c)
    T = promote_type(eltype(c),eltype(A),eltype(B),eltype(C),eltype(X))
    @boundscheck check_clenshaw_recurrences(N, A, B, C)
    m = size(X,1)
    m == size(X,2) || throw(DimensionMismatch("X must be square"))
    N == 0 && return zero(T)
    bn2 = _clenshaw_op(MemoryLayout(X), Zeros{T}(m, m), N)
    bn1 = _clenshaw_op(MemoryLayout(X), c[N]*Eye{T}(m), N)
    _clenshaw_op!(c, A, B, C, X, bn1, bn2)
end

function clenshaw(c::AbstractVector, A::AbstractVector, B::AbstractVector, C::AbstractVector, X::AbstractMatrix, f::AbstractVector)
    N = length(c)
    T = promote_type(eltype(c),eltype(A),eltype(B),eltype(C),eltype(X))
    @boundscheck check_clenshaw_recurrences(N, A, B, C)
    m = size(X,1)
    m == size(X,2) || throw(DimensionMismatch("X must be square"))
    m == length(f) || throw(DimensionMismatch("Dimensions must match"))
    N == 0 && return zero(T)
    bn2 = zeros(T,m)
    bn1 = Vector{T}(undef,m)
    bn1 .= c[N] .* f
    _clenshaw_op!(c, A, B, C, X, f, bn1, bn2)
end

function _clenshaw_op!(c, A, B, C, X, bn1, bn2)
    N = length(c)
    N == 1 && return bn1
    @inbounds begin
        for n = N-1:-1:2
            bn1,bn2 = _clenshaw_next!(n, A, B, C, X, c, bn1, bn2),bn1
        end
        bn1 = _clenshaw_first!(A, B, C, X, c, bn1, bn2)
    end
    bn1
end

function _clenshaw_op!(c, A, B, C, X, f::AbstractVector, bn1, bn2)
    N = length(c)
    N == 1 && return bn1
    @inbounds begin
        for n = N-1:-1:2
            bn1,bn2 = _clenshaw_next!(n, A, B, C, X, c, f, bn1, bn2),bn1
        end
        bn1 = _clenshaw_first!(A, B, C, X, c, f, bn1, bn2)
    end
    bn1
end



"""
    Clenshaw(a, X)

represents the operator `a(X)` where a is a polynomial.
Here `a` is to stored as a quasi-vector.
"""
struct Clenshaw{T, Coefs<:AbstractVector, AA<:AbstractVector, BB<:AbstractVector, CC<:AbstractVector, Jac<:AbstractMatrix} <: AbstractBandedMatrix{T}
    c::Coefs
    A::AA
    B::BB
    C::CC
    X::Jac
end

Clenshaw(c::AbstractVector{T}, A::AbstractVector, B::AbstractVector, C::AbstractVector, X::AbstractMatrix{T}) where T = 
    Clenshaw{T,typeof(c),typeof(A),typeof(B),typeof(C),typeof(X)}(c, A, B, C, X)

function Clenshaw(a::AbstractQuasiVector, X::AbstractQuasiMatrix)
    P,c = arguments(a)
    Clenshaw(paddeddata(c), recurrencecoefficients(P)..., jacobimatrix(X))
end

copy(M::Clenshaw) = M
size(M::Clenshaw) = size(M.X)
axes(M::Clenshaw) = axes(M.X)
bandwidths(M::Clenshaw) = (length(M.c)-1,length(M.c)-1)

function getindex(M::Clenshaw, kr::AbstractUnitRange, jr::AbstractUnitRange)
    b = bandwidth(M,1)
    jkr=max(1,min(jr[1],kr[1])-b÷2):max(jr[end],kr[end])+b÷2
    # relationship between jkr and kr, jr
    kr2,jr2 = kr.-jkr[1].+1,jr.-jkr[1].+1
    clenshaw(M.c, M.A, M.B, M.C, M.X[jkr, jkr])[kr2,jr2]
end

function getindex(M::Clenshaw{T}, kr::AbstractUnitRange, j::Integer) where T
    b = bandwidth(M,1)
    jkr=max(1,min(j,kr[1])-b÷2):max(j,kr[end])+b÷2
    # relationship between jkr and kr, jr
    kr2,j2 = kr.-jkr[1].+1,j-jkr[1]+1
    f = [Zeros{T}(j2-1); one(T); Zeros{T}(length(jkr)-j2)]
    clenshaw(M.c, M.A, M.B, M.C, M.X[jkr, jkr], f)[kr2]
end

getindex(M::Clenshaw, k::Int, j::Int) = M[k:k,j][1]

struct ClenshawLayout <: AbstractBandedLayout end
MemoryLayout(::Type{<:Clenshaw}) = ClenshawLayout()
# sublayout(::ClenshawLayout, ::Type{NTuple{2,OneTo{Int}}}) = ClenshawLayout()
# sub_materialize(::ClenshawLayout, V) = BandedMatrix(V)

function materialize!(M::MatMulVecAdd{<:ClenshawLayout,<:PaddedLayout,<:PaddedLayout})
    α,A,x,β,y = M.α,M.A,M.B,M.β,M.C
    length(y) == size(A,1) || throw(DimensionMismatch("Dimensions must match"))
    length(x) == size(A,2) || throw(DimensionMismatch("Dimensions must match"))
    x̃ = paddeddata(x)
    m = length(x̃)
    b = bandwidth(A,1)
    jkr=1:m+b÷2
    p = [x̃; zeros(eltype(x̃),length(jkr)-m)]
    Ax = clenshaw(A.c, A.A, A.B, A.C, A.X[jkr, jkr], p)
    v = view(y,jkr)
    v .+= α .* Ax .+ β .* v
    y
end

function \(A::OrthogonalPolynomial, B::BroadcastQuasiMatrix{<:Any, typeof(*), <:Tuple{<:Expansion{<:Any,<:OrthogonalPolynomial}, <:OrthogonalPolynomial}})
    a,P = arguments(B)
    (A \ P) * Clenshaw(a, P)
end