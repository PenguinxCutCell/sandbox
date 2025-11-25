using CartesianGeometry
using ImplicitCutIntegration
using Test

const T = Float64

body(x,y,z,w) = x^2 + y^2 + z^2 + w^2 - 1.0
mesh = (collect(-1:0.5:1), collect(-1:0.5:1), collect(-1:0.5:1), collect(-1:0.5:1))

# CartesianGeometry API
println("CartesianGeometry API")
Vs, bary, interface_hyperarea, cell_types = integrate(Tuple{0}, body, mesh, T, zero; method=:vofijul)
As = integrate(Tuple{1}, body, mesh, T, zero; method=:vofijul)
Ws = integrate(Tuple{0}, body, mesh, T, zero, bary; method=:vofijul)
Bs = integrate(Tuple{1}, body, mesh, T, zero, bary; method=:vofijul)

# ImplicitCutIntegration API
println("ImplicitCutIntegration API")
A, B, V, W, C_ω, C_γ, Γ, cell_types = GeometricMoments(body, mesh; compute_centroids = false)

@testset "GeometricMoments Size" begin
    @test all(size(Vs) .== size(V))
    @test all(size(As[1]) .== size(A[1]))
    @test all(size(Ws[1]) .== size(W[1]))
    @test all(size(Bs[1]) .== size(B[1]))
end


# helper: convert sparse/diagonal matrix or vector to an array shaped like `ref`
function sparsediag_to_array(x, ref)
    if isa(x, AbstractMatrix)
        n = prod(size(ref))
        # collect diagonal entries (or zeros if matrix smaller)
        d = [i <= size(x,1) && i <= size(x,2) ? x[i,i] : zero(eltype(x)) for i in 1:n]
        return reshape(d, size(ref))
    else
        # already vector/array: reshape to ref shape if necessary
        return reshape(collect(x), size(ref))
    end
end

# convert GeometricMoments outputs to dense arrays matching CartesianGeometry outputs
V_arr  = sparsediag_to_array(V,  Vs)
A1_arr = sparsediag_to_array(A[1], As[1])
W1_arr = sparsediag_to_array(W[1], Ws[1])
B1_arr = sparsediag_to_array(B[1], Bs[1])

@testset "GeometricMoments Values" begin
    @test isapprox(Vs,  V_arr;  atol=1e-3)
    @test isapprox(As[1], A1_arr; atol=1e-3)
    @test isapprox(Ws[1], W1_arr; atol=1e-3)
    @test isapprox(Bs[1], B1_arr; atol=1e-3)
end