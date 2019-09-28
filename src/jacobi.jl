abstract type AbstractJacobiWeight{T} <: AbstractQuasiVector{T} end

struct JacobiWeight{T} <: AbstractJacobiWeight{T}
    b::T
    a::T
    JacobiWeight{T}(b, a) where T = new{T}(convert(T,b), convert(T,a))
end

JacobiWeight(b::T, a::V) where {T,V} = JacobiWeight{promote_type(T,V)}(b,a)

axes(::AbstractJacobiWeight) = (Inclusion(ChebyshevInterval()),)
function getindex(w::JacobiWeight, x::Number)
    x ∈ axes(w,1) || throw(BoundsError())
    (1-x)^w.a * (1+x)^w.b
end



abstract type AbstractJacobi{T} <: OrthogonalPolynomial{T} end

struct Legendre{T} <: AbstractJacobi{T} end
Legendre() = Legendre{Float64}()

==(::Legendre, ::Legendre) = true

struct Jacobi{T} <: AbstractJacobi{T}
    b::T
    a::T
    Jacobi{T}(b, a) where T = new{T}(convert(T,b), convert(T,a))
end

Jacobi(b::T, a::V) where {T,V} = Jacobi{promote_type(T,V)}(b,a)

axes(::AbstractJacobi) = (Inclusion(ChebyshevInterval()), OneTo(∞))
==(P::Jacobi, Q::Jacobi) = P.a == Q.a && P.b == Q.b

########
# Mass Matrix
#########

@simplify *(A::QuasiAdjoint{<:Any,<:Legendre}, B::Legendre) =
    Diagonal(2 ./ (2(0:∞) .+ 1))

########
# Jacobi Matrix
########

jacobimatrix(::Legendre) = _BandedMatrix(Vcat(((0:∞)./(1:2:∞))', Zeros(1,∞), ((1:∞)./(1:2:∞))'), ∞, 1,1)

function jacobimatrix(J::Jacobi) 
    b,a = J.b,J.a
    n = 0:∞
    B = @. 2*(n+1)*(n+a+b+1) / ((2n+a+b+1)*(2n+a+b+2))
    A = Vcat((a-b) / (a+b+2), (a^2-b^2) ./ ((2n.+a.+b.+2).*(2n.+a.+b.+4)))
    C = @. 2*(n+a)*(n+b) / ((2n+a+b)*(2n+a+b+1))

    _BandedMatrix(Vcat(C',A',B'), ∞, 1,1)
end


@simplify *(X::Identity, P::Legendre) = ApplyQuasiMatrix(*, P, P\(X*P))



##########
# Conversion
##########

@simplify function *(A::Jacobi, B::Jacobi) 
    a,b = B.a,B.b
    if A.a == a && A.b == b+1
        _BandedMatrix(Vcat((((0:∞) .+ a)./((1:2:∞) .+ (a+b)))', (((1:∞) .+ (a+b))./((1:2:∞) .+ (a+b)))'), ∞, 0,1)
    end
end

##########
# Derivatives
##########

# Jacobi(b+1,a+1)\(D*Jacobi(a,b))
@simplify function *(D::Derivative{<:Any,<:ChebyshevInterval}, S::Jacobi)
    A = _BandedMatrix((((-1:∞) .+ (S.a + S.b))/2)', ∞, -1,1)
    ApplyQuasiMatrix(*, Jacobi(S.b+1,S.a+1), A)
end

# Legendre()\ (D*W*Jacobi(true,true))
@simplify function *(D::Derivative{<:Any,<:ChebyshevInterval}, WS::WeightedBasis{Bool,JacobiWeight{Bool},Jacobi{Bool}})
    w,S = WS.args                    
    (w.a && S.a && w.b && S.b) || throw(ArgumentError())
    A = _BandedMatrix((-2*(1:∞))', ∞, 1,-1)
    ApplyQuasiMatrix(*, Legendre(), A)
end

# Jacobi(b-1,a-1)\ (D*w*Jacobi(b,a))
@simplify function *(D::Derivative{<:Any,<:ChebyshevInterval}, WS::WeightedBasis{<:Any,<:JacobiWeight,<:Jacobi})
    w,S = WS.args
    a,b = S.a, S.b
    (w.a == a && w.b == b) || throw(ArgumentError())
    A = _BandedMatrix((-2*(1:∞))', ∞, 1,-1)
    ApplyQuasiMatrix(*, Jacobi(a-1,b-1), A)
end

@simplify function \(J::Jacobi{Bool}, WS::WeightedBasis{Bool,JacobiWeight{Bool},Jacobi{Bool}})
    w,S = WS.args
    @assert  S.b && S.a
    if w.b && !w.a
        @assert !J.b && J.a
        _BandedMatrix(Vcat(((2:2:∞)./(3:2:∞))',((2:2:∞)./(3:2:∞))'), ∞, 1,0)
    elseif !w.b && w.a
        @assert J.b && !J.a
        _BandedMatrix(Vcat(((2:2:∞)./(3:2:∞))',(-(2:2:∞)./(3:2:∞))'), ∞, 1,0)
    else
        error("Not implemented")
    end
end

@simplify function \(L::Legendre, WS::WeightedBasis{Bool,JacobiWeight{Bool},Jacobi{Bool}})
    w,S = WS.args
    if w.b && w.a
        @assert S.b && S.a
        _BandedMatrix(Vcat(((2:2:∞)./(3:2:∞))', Zeros(1,∞), (-(2:2:∞)./(3:2:∞))'), ∞, 2,0)
    elseif w.b && !w.a
        @assert S.b && !S.a
        _BandedMatrix(Ones{eltype(M)}(2,∞), ∞, 1,0)
    elseif !w.b && w.a
        @assert !S.b && S.a
        _BandedMatrix(Vcat(Ones{eltype(M)}(1,∞),-Ones{eltype(M)}(1,∞)), ∞, 1,0)
    else
        error("Not implemented")
    end
end

@simplify function *(St::QuasiAdjoint{Bool,Jacobi{Bool}}, WS::WeightedBasis{Int,JacobiWeight{Int},Jacobi{Bool}})
    w = parent(W)
    (w.b == 2 && S.b && w.a == 2 && S.a && parent(St) == S) || throw(ArgumentError())
    W_sqrt = Diagonal(JacobiWeight(true,true))
    L = Legendre()
    A = PInv(L)*W_sqrt*S
    A'*(L'L)*A
end
