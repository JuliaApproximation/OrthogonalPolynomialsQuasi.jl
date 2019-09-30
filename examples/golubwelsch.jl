using OrthogonalPolynomialsQuasi, FastGaussQuadrature

P = Legendre()
x = axes(P,1)
x .* P