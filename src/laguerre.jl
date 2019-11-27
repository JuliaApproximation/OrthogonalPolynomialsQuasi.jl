struct Laguerre{T} <: OrthogonalPolynomial{T} 
    α::T
    Laguerre{T}(α) where T = new{T}(convert(T, α))
end
Laguerre{T}() where T = Laguerre{T}(zero(T))
Laguerre() = Laguerre{Float64}()
Laguerre(α::T) where T = Laguerre{float(T)}(α)

==(L1::Laguerre, L2::Laguerre) = L1.α == L2.α
axes(::Laguerre{T}) where T = (Inclusion(HalfLine{T}()), OneTo(∞))

# L_{n+1} = (-1/(n+1) x + (2n+α+1)/(n+1)) L_n - (n+α)/(n+1) L_{n-1}
# - (n+α) L_{n-1} + (2n+α+1)* L_n -(n+1) L_{n+1} = x  L_n
# x*[L_0 L_1 L_2 …] = [L_0 L_1 L_2 …] * [(α+1)    -(α+1); -1  (α+3)     -(α+2);0  -2   (α+5) -(α+3); …]   
function jacobimatrix(L::Laguerre{T}) where T
    α = L.α
    _BandedMatrix(Vcat((-(α:∞))', ((α+1):2:∞)', (-(1:∞))'), ∞, 1, 1)
end

##########
# Derivatives
##########

@simplify function *(D::Derivative, L::Laguerre)
    T = promote_type(eltype(D),eltype(L))
    D = _BandedMatrix(Fill(-one(T),1,∞), ∞, -1,1)
    Laguerre(L.α+1)*D
end