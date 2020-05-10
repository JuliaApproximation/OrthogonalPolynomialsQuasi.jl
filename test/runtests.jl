using Base, OrthogonalPolynomialsQuasi, ContinuumArrays, QuasiArrays, FillArrays,
        LazyArrays, BandedMatrices, LinearAlgebra, FastTransforms, IntervalSets,
        InfiniteLinearAlgebra, Test
using ForwardDiff, SemiseparableMatrices, SpecialFunctions, LazyBandedMatrices
import ContinuumArrays: SimplifyStyle, BasisLayout, MappedBasisLayout
import OrthogonalPolynomialsQuasi: jacobimatrix, ∞, ChebyshevInterval
import LazyArrays: ApplyStyle, colsupport, MemoryLayout, arguments
import SemiseparableMatrices: VcatAlmostBandedLayout
import QuasiArrays: MulQuasiMatrix
import Base: OneTo
import InfiniteLinearAlgebra: KronTrav, Block

include("test_chebyshev.jl")
include("test_legendre.jl")
include("test_ultraspherical.jl")
include("test_jacobi.jl")
include("test_odes.jl")

@testset "Auto-diff" begin
    U = Ultraspherical(1)
    C = Ultraspherical(2)

    f = x -> ChebyshevT{eltype(x)}()[x,5]
    @test ForwardDiff.derivative(f,0.1) ≈ 4*U[0.1,4]
    f = x -> ChebyshevT{eltype(x)}()[x,5][1]
    @test ForwardDiff.gradient(f,[0.1]) ≈ [4*U[0.1,4]]
    @test ForwardDiff.hessian(f,[0.1]) ≈ [8*C[0.1,3]]

    f = x -> ChebyshevT{eltype(x)}()[x,1:5]
    @test ForwardDiff.derivative(f,0.1) ≈ [0;(1:4).*U[0.1,1:4]]
end