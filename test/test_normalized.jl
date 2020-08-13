using OrthogonalPolynomialsQuasi
import OrthogonalPolynomialsQuasi: NormalizationConstant, recurrencecoefficients, Normalized

P = Legendre()
Q = Normalized(P)

h = NormalizationConstant(P)
A, B, C = recurrencecoefficients(P)
(h[2:∞] ./ h) .* A
(h[2:∞] ./ h) .* B
(h[3:∞] ./ h) .* C

P'P

@ent sum(P; dims=1)

@ent sum(P * [1; zeros(∞)])

Diagonal(NormalizationConstant(P))