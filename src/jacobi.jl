abstract type AbstractJacobiWeight{T} <: Weight{T} end

struct JacobiWeight{T} <: AbstractJacobiWeight{T}
    b::T
    a::T
    JacobiWeight{T}(b, a) where T = new{T}(convert(T,b), convert(T,a))
end

JacobiWeight(b::T, a::V) where {T,V} = JacobiWeight{promote_type(T,V)}(b,a)

==(A::JacobiWeight, B::JacobiWeight) = A.b == B.b && A.a == B.a

axes(::AbstractJacobiWeight{T}) where T = (Inclusion(ChebyshevInterval{T}()),)
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

Jacobi(P::Legendre{T}) where T = Jacobi(zero(T), zero(T))

axes(::AbstractJacobi{T}) where T = (Inclusion(ChebyshevInterval{T}()), OneTo(∞))
==(P::Jacobi, Q::Jacobi) = P.a == Q.a && P.b == Q.b
==(P::Legendre, Q::Jacobi) = Jacobi(P) == Q
==(P::Jacobi, Q::Legendre) = P == Jacobi(Q)
==(A::WeightedBasis{<:Any,<:JacobiWeight,<:Jacobi}, B::WeightedBasis{<:Any,<:JacobiWeight,<:Jacobi}) = 
    A.args == B.args
==(A::WeightedBasis{<:Any,<:JacobiWeight,<:Jacobi}, B::Jacobi{T}) where T = 
    A == JacobiWeight(zero(T),zero(T)).*B
==(A::WeightedBasis{<:Any,<:JacobiWeight,<:Jacobi}, B::Legendre) = 
    A == Jacobi(B)
==(A::Jacobi{T}, B::WeightedBasis{<:Any,<:JacobiWeight,<:Jacobi}) where T = 
    JacobiWeight(zero(T),zero(T)).*A == B
==(A::Legendre, B::WeightedBasis{<:Any,<:JacobiWeight,<:Jacobi}) =     
    Jacobi(A) == B

###
# transforms
###

function grid(Tn::SubQuasiArray{<:Any,2,<:AbstractJacobi,<:Tuple{<:Inclusion,<:AbstractUnitRange}}) 
    kr,jr = parentindices(Tn)
    ChebyshevGrid{1,eltype(kr)}(maximum(jr))
end    

########
# Mass Matrix
#########

@simplify *(A::QuasiAdjoint{<:Any,<:Legendre}, B::Legendre) =
    Diagonal(2 ./ (2(0:∞) .+ 1))

function legendre_massmatrix(Ac, B)
    A = parent(Ac)
    P = Legendre{eltype(B)}()
    (P\A)'*(P'P)*(P\B)
end

@simplify *(Ac::QuasiAdjoint{<:Any,<:Jacobi}, B::Jacobi) = legendre_massmatrix(Ac,B)
@simplify *(Ac::QuasiAdjoint{<:Any,<:WeightedBasis{<:Any,<:JacobiWeight}}, B::WeightedBasis{<:Any,<:JacobiWeight}) = legendre_massmatrix(Ac,B)
@simplify *(Ac::QuasiAdjoint{<:Any,<:Jacobi}, B::WeightedBasis{<:Any,<:JacobiWeight})  = legendre_massmatrix(Ac,B)

########
# Jacobi Matrix
########

jacobimatrix(::Legendre) = _BandedMatrix(Vcat(((0:∞)./(1:2:∞))', Zeros(1,∞), ((1:∞)./(1:2:∞))'), ∞, 1,1)

function jacobimatrix(J::Jacobi) 
    b,a = J.b,J.a
    n = 0:∞
    B = @. 2*(n+1)*(n+a+b+1) / ((2n+a+b+1)*(2n+a+b+2))
    A = Vcat((b-a) / (a+b+2), (b^2-a^2) ./ ((2n.+a.+b.+2).*(2n.+a.+b.+4)))
    C = @. 2*(n+a)*(n+b) / ((2n+a+b)*(2n+a+b+1))

    _BandedMatrix(Vcat(C',A',B'), ∞, 1,1)
end


@simplify *(X::Identity, P::Legendre) = ApplyQuasiMatrix(*, P, P\(X*P))



##########
# Conversion
##########

@simplify \(A::Jacobi, B::Legendre) = A\Jacobi(B)
@simplify \(A::Legendre, B::Jacobi) = Jacobi(A)\B

@simplify function \(A::Jacobi, B::Jacobi) 
    T = promote_type(eltype(A), eltype(B))
    a,b = B.a,B.b
    if A.a == a && A.b == b
        Eye{T}(∞)
    elseif A.a == a && A.b == b+1
        _BandedMatrix(Vcat((((0:∞) .+ a)./((1:2:∞) .+ (a+b)))', (((1:∞) .+ (a+b))./((1:2:∞) .+ (a+b)))'), ∞, 0,1)
    else
        error("not implemented for $A and $B")
    end
end

@simplify function \(A::Jacobi, w_B::WeightedBasis{<:Any,<:JacobiWeight,<:Jacobi}) 
    a,b = A.a,A.b
    (JacobiWeight(zero(a),zero(b)) .* A) \ w_B
end

@simplify function \(w_A::WeightedBasis{<:Any,<:JacobiWeight,<:Jacobi}, B::Jacobi) 
    a,b = B.a,B.b
    w_A \ (JacobiWeight(zero(a),zero(b)) .* B)
end

@simplify function \(w_A::WeightedBasis{<:Any,<:JacobiWeight,<:Jacobi}, w_B::WeightedBasis{<:Any,<:JacobiWeight,<:Jacobi}) 
    wA,A = w_A.args
    wB,B = w_B.args

    if wA == wB
        A \ B
    elseif B.a == A.a && B.b == A.b+1 && wB.b == wA.b+1 && wB.a == wA.a
        _BandedMatrix(Vcat((((2:2:∞) .+ 2A.b)./((2:2:∞) .+ (A.a+A.b)))', ((2:2:∞)./((2:2:∞) .+ (A.a+A.b)))'), ∞, 1,0)
    elseif B.a == A.a+1 && B.b == A.b && wB.b == wA.b && wB.a == wA.a+1
        _BandedMatrix(Vcat((((2:2:∞) .+ 2A.a)./((2:2:∞) .+ (A.a+A.b)))', -((2:2:∞)./((2:2:∞) .+ (A.a+A.b)))'), ∞, 1,0)
    elseif wB.a ≥ wA.a+1
        J = JacobiWeight(wB.b,wB.a-1) .* Jacobi(B.b,B.a-1) 
        (w_A\J) * (J\w_B)
    elseif wB.b ≥ wA.b+1
        J = JacobiWeight(wB.b-1,wB.a) .* Jacobi(B.b-1,B.a) 
        (w_A\J) * (J\w_B)
    else
        error("not implemented for $A and $wB")
    end
end

\(A::Legendre, wB::WeightedBasis{<:Any,<:JacobiWeight,<:Jacobi}) = Jacobi(A) \ wB

##########
# Derivatives
##########

# Jacobi(b+1,a+1)\(D*Jacobi(a,b))
@simplify function *(D::Derivative{<:Any,<:AbstractInterval}, S::Jacobi)
    A = _BandedMatrix((((1:∞) .+ (S.a + S.b))/2)', ∞, -1,1)
    ApplyQuasiMatrix(*, Jacobi(S.b+1,S.a+1), A)
end

# Jacobi(b-1,a-1)\ (D*w*Jacobi(b,a))
@simplify function *(D::Derivative{<:Any,<:AbstractInterval}, WS::WeightedBasis{<:Any,<:JacobiWeight,<:Jacobi})
    w,S = WS.args
    a,b = S.a, S.b
    (w.a == a && w.b == b) || throw(ArgumentError())
    if w.a == 0 && w.b == 0
        D*S
    else
        A = _BandedMatrix((-2*(1:∞))', ∞, 1,-1)
        ApplyQuasiMatrix(*, JacobiWeight(a-1,b-1) .* Jacobi(a-1,b-1), A)
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


###
# Splines
###

@simplify function \(A::Legendre, B::HeavisideSpline)
    @assert B.points == -1:2:1
    Vcat(1, Zeros(∞,1))
end