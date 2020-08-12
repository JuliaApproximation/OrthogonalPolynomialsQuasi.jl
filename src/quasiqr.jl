struct Weighted{T, BASIS} <: Basis{T}

end



struct SqrtWeighted{T, BASIS} <: Basis{T}

end

struct Normalized{T, POLY<:OrthogonalPolynomial{T}} <: OrthogonalPolynomial{T}
    polynomials::POLY
end

struct OrthogonalPolynomialQR{T, POLY}

end

qr(A::OrthogonalPolynomial) = 