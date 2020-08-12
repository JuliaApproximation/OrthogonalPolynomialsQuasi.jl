using OrthogonalPolynomialsQuasi
import OrthogonalPolynomialsQuasi: NormalizationConstant

P = Legendre()
P'P

@ent sum(P; dims=1)

@ent sum(P * [1; zeros(∞)])

Diagonal(NormalizationConstant(P))