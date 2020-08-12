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

const WeightedJacobi{T} = WeightedBasis{T,<:JacobiWeight,<:Jacobi}

WeightedJacobi(b,a) = JacobiWeight(b,a) .* Jacobi(b,a)
WeightedJacobi{T}(b,a) where T = JacobiWeight{T}(b,a) .* Jacobi{T}(b,a)


axes(::AbstractJacobi{T}) where T = (Inclusion(ChebyshevInterval{T}()), OneTo(∞))
==(P::Jacobi, Q::Jacobi) = P.a == Q.a && P.b == Q.b
==(P::Legendre, Q::Jacobi) = Jacobi(P) == Q
==(P::Jacobi, Q::Legendre) = P == Jacobi(Q)
==(A::WeightedJacobi, B::WeightedJacobi) = 
    A.args == B.args
==(A::WeightedJacobi, B::Jacobi{T}) where T = 
    A == JacobiWeight(zero(T),zero(T)).*B
==(A::WeightedJacobi, B::Legendre) = 
    A == Jacobi(B)
==(A::Jacobi{T}, B::WeightedJacobi) where T = 
    JacobiWeight(zero(T),zero(T)).*A == B
==(A::Legendre, B::WeightedJacobi) =     
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

# 2^{a + b + 1} {\Gamma(n+a+1) \Gamma(n+b+1) \over (2n+a+b+1) \Gamma(n+a+b+1) n!}.

function jacobi_massmatrix(b, a) 
    n = 0:∞
    Diagonal(2^(a+b+1) * (@. exp(loggamma(n+a+1) + loggamma(n+b+1) - loggamma(n+a+b+1) - loggamma(n+1)) / (2n+a+b+1)))
end

@simplify *(Ac::QuasiAdjoint{<:Any,<:AbstractJacobi}, B::AbstractJacobi) = legendre_massmatrix(Ac,B)
@simplify *(Ac::QuasiAdjoint{<:Any,<:WeightedBasis{<:Any,<:AbstractJacobiWeight}}, B::WeightedBasis{<:Any,<:AbstractJacobiWeight}) = legendre_massmatrix(Ac,B)
@simplify *(Ac::QuasiAdjoint{<:Any,<:AbstractJacobi}, B::WeightedBasis{<:Any,<:AbstractJacobiWeight})  = legendre_massmatrix(Ac,B)

########
# Jacobi Matrix
########

jacobimatrix(::Legendre) = _BandedMatrix(Vcat(((0:∞)./(1:2:∞))', Zeros(1,∞), ((1:∞)./(1:2:∞))'), ∞, 1,1)

# These return vectors A[k], B[k], C[k] are from DLMF. Cause of MikaelSlevinsky we need an extra entry in C ... for now.
function recurrencecoefficients(::Legendre{T}) where T
    n = zero(T):∞
    ((2n .+ 1) ./ (n .+ 1), Zeros{T}(∞), n ./ (n .+ 1))
end

function jacobimatrix(J::Jacobi) 
    b,a = J.b,J.a
    n = 0:∞
    B = Vcat(2 / (a+b+2),  @. 2*(n+2)*(n+a+b+2) / ((2n+a+b+3)*(2n+a+b+4)))
    A = Vcat((b-a) / (a+b+2), (b^2-a^2) ./ ((2n.+a.+b.+2).*(2n.+a.+b.+4)))
    C = @. 2*(n+a)*(n+b) / ((2n+a+b)*(2n+a+b+1))

    _BandedMatrix(Vcat(C',A',B'), ∞, 1,1)
end

function recurrencecoefficients(P::Jacobi)
    n = 0:∞
    ñ = 1:∞
    a,b = P.a,P.b
    A = Vcat((a+b+2)/2, (2ñ .+ (a+b+1)) .* (2ñ .+ (a+b+2)) ./ ((2*(ñ .+ 1)) .* (ñ .+ (a+b+1))))
    # n = 0 is special to avoid divide-by-zero
    B = Vcat((a-b)/2, (a^2 - b^2) * (2ñ .+ (a + b+1)) ./ ((2*(ñ .+ 1)) .* (ñ .+ (a+b+1)) .* (2ñ .+ (a+b))))
    C = ((n .+ a) .* (n .+ b) .* (2n .+ (a+b+2))) ./ (ñ .* (n .+ (a +b+1)) .* (2n .+ (a+b)))
    (A,B,C)
end


@simplify *(X::Identity, P::Legendre) = ApplyQuasiMatrix(*, P, P\(X*P))



##########
# Conversion
##########

\(A::Jacobi, B::Legendre) = A\Jacobi(B)
\(A::Legendre, B::Jacobi) = Jacobi(A)\B

function \(A::Jacobi, B::Jacobi) 
    T = promote_type(eltype(A), eltype(B))
    a,b = B.a,B.b
    if A.a == a && A.b == b
        Eye{T}(∞)
    elseif isone(-a-b) && A.a == a && A.b == b+1
        _BandedMatrix(Vcat((((0:∞) .+ a)./((1:2:∞) .+ (a+b)))', 
                            Vcat(1,((2:∞) .+ (a+b))./((3:2:∞) .+ (a+b)))'), ∞, 0,1)
    elseif isone(-a-b) && A.a == a+1 && A.b == b
        _BandedMatrix(Vcat((-((0:∞) .+ b)./((1:2:∞) .+ (a+b)))', 
                            Vcat(1,((2:∞) .+ (a+b))./((3:2:∞) .+ (a+b)))'), ∞, 0,1)
    elseif A.a == a && A.b == b+1
        _BandedMatrix(Vcat((((0:∞) .+ a)./((1:2:∞) .+ (a+b)))', 
                            (((1:∞) .+ (a+b))./((1:2:∞) .+ (a+b)))'), ∞, 0,1)
    elseif A.a == a+1 && A.b == b
        _BandedMatrix(Vcat((-((0:∞) .+ b)./((1:2:∞) .+ (a+b)))', 
                            (((1:∞) .+ (a+b))./((1:2:∞) .+ (a+b)))'), ∞, 0,1)
    elseif A.a ≥ a+1
        J = Jacobi(b,a+1)
        (A \ J) * (J \ B)
    elseif A.b ≥ b+1
        J = Jacobi(b+1,a)
        (A \ J) * (J \ B)
    else        
        error("not implemented for $A and $B")
    end
end

function \(A::Jacobi, w_B::WeightedJacobi) 
    a,b = A.a,A.b
    (JacobiWeight(zero(a),zero(b)) .* A) \ w_B
end

function \(w_A::WeightedJacobi, B::Jacobi) 
    a,b = B.a,B.b
    w_A \ (JacobiWeight(zero(a),zero(b)) .* B)
end

function \(w_A::WeightedJacobi, w_B::WeightedJacobi) 
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

\(A::Legendre, wB::WeightedJacobi) = Jacobi(A) \ wB

##########
# Derivatives
##########

# Jacobi(b+1,a+1)\(D*Jacobi(b,a))
@simplify function *(D::Derivative{<:Any,<:AbstractInterval}, S::Jacobi)
    A = _BandedMatrix((((1:∞) .+ (S.a + S.b))/2)', ∞, -1,1)
    ApplyQuasiMatrix(*, Jacobi(S.b+1,S.a+1), A)
end

# Jacobi(b-1,a-1)\ (D*w*Jacobi(b,a))
@simplify function *(D::Derivative{<:Any,<:AbstractInterval}, WS::WeightedJacobi)
    w,S = WS.args
    a,b = S.a, S.b
    if w.a == 0 && w.b == 0
        D*S
    elseif iszero(w.a) && w.b == b #L_6
        A = _BandedMatrix((b:∞)', ∞, 0,0)
        ApplyQuasiMatrix(*, JacobiWeight(b-1,w.a) .* Jacobi(b-1,a+1), A)
    elseif iszero(w.b) && w.a == a #L_6^t
        A = _BandedMatrix((a:∞)', ∞, 0,0)
        ApplyQuasiMatrix(*, JacobiWeight(w.b,a-1) .* Jacobi(b+1,a-1), A)        
    elseif w.a == a && w.b == b # L_1^t
        A = _BandedMatrix((-2*(1:∞))', ∞, 1,-1)
        ApplyQuasiMatrix(*, JacobiWeight(b-1,a-1) .* Jacobi(b-1,a-1), A)    
    elseif iszero(w.a)
        W = (JacobiWeight(b-1,w.a) .* Jacobi(b-1,a+1)) \ (D * (JacobiWeight(b,w.a) .* S))
        J = Jacobi(b,a+1) # range Jacobi
        C1 = J \ Jacobi(b-1,a+1)
        C2 = J \ Jacobi(b,a)
        ApplyQuasiMatrix(*, JacobiWeight(w.b-1,w.a) .* J, (w.b-b) * C2 + C1 * W)
    else
        error("Not implemented")
    end
end


function \(L::Legendre, WS::WeightedBasis{Bool,JacobiWeight{Bool},Jacobi{Bool}})
    w,S = WS.args
    if w.b && w.a
        @assert S.b && S.a
        _BandedMatrix(Vcat(((2:2:∞)./(3:2:∞))', Zeros(1,∞), (-(2:2:∞)./(3:2:∞))'), ∞, 2,0)
    elseif w.b && !w.a
        @assert S.b && !S.a
        _BandedMatrix(Ones{eltype(L)}(2,∞), ∞, 1,0)
    elseif !w.b && w.a
        @assert !S.b && S.a
        _BandedMatrix(Vcat(Ones{eltype(L)}(1,∞),-Ones{eltype(L)}(1,∞)), ∞, 1,0)
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

function \(A::Legendre, B::HeavisideSpline)
    @assert B.points == -1:2:1
    Vcat(1, Zeros(∞,1))
end

###
# sum
###

function _sum(P::Legendre{T}, dims) where T
    @assert dims == 1
    Hcat(convert(T, 2), Zeros{T}(1,∞))
end

_sum(p::SubQuasiArray{T,1,Legendre{T},<:Tuple{Inclusion,Int}}, ::Colon) where T = parentindices(p)[2] == 1 ? convert(T, 2) : zero(T)