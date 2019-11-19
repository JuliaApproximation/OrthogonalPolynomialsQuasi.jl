##
# Chebyshev
##

struct ChebyshevWeight{kind,T} <: AbstractJacobiWeight{T} end
ChebyshevWeight{kind}() where kind = ChebyshevWeight{kind,Float64}()
ChebyshevWeight() = ChebyshevWeight{1,Float64}()

struct Chebyshev{kind,T} <: AbstractJacobi{T} end
Chebyshev() = Chebyshev{1,Float64}()
Chebyshev{kind}() where kind = Chebyshev{kind,Float64}()

const ChebyshevTWeight = ChebyshevWeight{1}
const ChebyshevUWeight = ChebyshevWeight{2}
const ChebyshevT = Chebyshev{1}
const ChebyshevU = Chebyshev{2}

==(a::Chebyshev{kind}, b::Chebyshev{kind}) where kind = true
==(a::Chebyshev, b::Chebyshev) = false


function getindex(w::ChebyshevTWeight, x::Number)
    x ∈ axes(w,1) || throw(BoundsError())
    inv(sqrt(1-x^2))
end

function getindex(w::ChebyshevUWeight, x::Number)
    x ∈ axes(w,1) || throw(BoundsError())
    sqrt(1-x^2)
end

struct ChebyshevGrid{kind,T} <: AbstractVector{T}
    n::Int
end

ChebyshevGrid{kind}(n::Integer) where kind = ChebyshevGrid{kind,Float64}(n)

size(g::ChebyshevGrid) = (g.n,)
getindex(g::ChebyshevGrid{1,T}, k::Integer) where T = 
    sinpi(convert(T,g.n-2k+1)/(2g.n))

function getindex(g::ChebyshevGrid{2,T}, k::Integer) where T
    g.n == 1 && return zero(T)
    sinpi(convert(T,g.n-2k+1)/(2g.n-2))
end



Jacobi(C::ChebyshevT{T}) where T = Jacobi(-one(T)/2,-one(T)/2)
Jacobi(C::ChebyshevU{T}) where T = Jacobi(one(T)/2,one(T)/2)

########
# Jacobi Matrix
########

jacobimatrix(C::ChebyshevT{T}) where T = 
    _BandedMatrix(Vcat(Fill(one(T)/2,1,∞), 
                        Zeros{T}(1,∞), 
                        Hcat(one(T), Fill(one(T)/2,1,∞))), ∞, 1, 1)

jacobimatrix(C::ChebyshevU{T}) where T = 
    _BandedMatrix(Vcat(Fill(one(T)/2,1,∞), 
                        Zeros{T}(1,∞), 
                        Fill(one(T)/2,1,∞)), ∞, 1, 1)   
                        
##########
# Derivatives
##########

# Ultraspherical(1)\(D*Chebyshev())
@simplify function *(D::Derivative{<:Any,<:ChebyshevInterval}, S::ChebyshevT)
    T = promote_type(eltype(D),eltype(S))
    A = _BandedMatrix((zero(T):∞)', ∞, -1,1)
    ApplyQuasiMatrix(*, ChebyshevU{T}(), A)
end                        

#####
# Conversion
#####

@simplify function \(U::ChebyshevU, C::ChebyshevT)
    T = promote_type(eltype(U), eltype(C))
    _BandedMatrix(Vcat(-Ones{T}(1,∞)/2,
                        Zeros{T}(1,∞), 
                        Hcat(Ones{T}(1,1),Ones{T}(1,∞)/2)), ∞, 0,2)
end

@simplify function \(w_A::WeightedBasis{<:Any,<:ChebyshevTWeight,<:ChebyshevT}, w_B::WeightedBasis{<:Any,<:ChebyshevUWeight,<:ChebyshevU}) 
    wA,A = w_A.args
    wB,B = w_B.args
    T = promote_type(eltype(w_A), eltype(w_B))
    _BandedMatrix(Vcat(Fill(one(T)/2, 1, ∞), Zeros{T}(1, ∞), Fill(-one(T)/2, 1, ∞)), ∞, 2, 0)
end


####
# interrelationships
####

# (18.7.3)

@simplify function \(A::ChebyshevT, B::Jacobi)
    T = promote_type(eltype(A), eltype(B))
    (B.a == B.b == -T/2) || throw(ArgumentError())
    Diagonal(Jacobi(-T/2,-T/2)[1,:])
end


####
# sum
####

function _sum(A::WeightedBasis{T,<:ChebyshevUWeight,<:ChebyshevU}, dims) where T
    w, U = A.args
    @assert dims == 1
    Hcat(convert(T, π)/2, Zeros{T}(1,∞))
end

function _sum(A::WeightedBasis{T,<:ChebyshevWeight,<:Chebyshev}, dims) where T 
    @assert dims == 1
    Hcat(convert(T, π), Zeros{T}(1,∞))
end