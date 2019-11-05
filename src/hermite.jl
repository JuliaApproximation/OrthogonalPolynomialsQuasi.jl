struct Hermite{T} <: OrthogonalPolynomial{T} end
Hermite() = Hermite{Float64}()

==(::Hermite, ::Hermite) = true
axes(::Hermite{T}) where T = (Inclusion(ℝ), OneTo(∞))

# H_{n+1} = 2x H_n - 2n H_{n-1}
# 1/2 * H_{n+1} + n H_{n-1} = x H_n 
# x*[H_0 H_1 H_2 …] = [H_0 H_1 H_2 …] * [0    1; 1/2  0     2; 1/2   0  3; …]   
function jacobimatrix(H::Hermite{T}) where T
    # X = BandedMatrix(1 => 1:∞, -1 => Fill(one(T)/2,∞))
    _BandedMatrix(Vcat((0:∞)', Zeros(1,∞), Fill(one(T)/2,1,∞)), ∞, 1, 1)
end

##########
# Derivatives
##########

@simplify function *(D::Derivative, H::Hermite)
    T = promote_type(eltype(D),eltype(H))
    D = _BandedMatrix((zero(T):2:∞)', ∞, -1,1)
    H*D
end