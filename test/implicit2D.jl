using CartesianGeometry
using ImplicitCutIntegration
using Test

const T = Float64

body(x,y) = x^2 + y^2 - 1.0
mesh = (collect(-1:0.5:1), collect(-1:0.5:1))

# CartesianGeometry API
println("CartesianGeometry API")
Vs, bary, interface_hyperarea, cell_types, bary_interface = integrate(Tuple{0}, body, mesh, T, zero; method=:vofijul)
As = integrate(Tuple{1}, body, mesh, T, zero; method=:vofijul)
Ws = integrate(Tuple{0}, body, mesh, T, zero, bary; method=:vofijul)
Bs = integrate(Tuple{1}, body, mesh, T, zero, bary; method=:vofijul)

# ImplicitCutIntegration API
println("ImplicitCutIntegration API")
A, B, V, W, C_ω, C_γ, Γ, cell_types = GeometricMoments(body, mesh; compute_centroids = true)

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
A2_arr = sparsediag_to_array(A[2], As[2])
W1_arr = sparsediag_to_array(W[1], Ws[1])
W2_arr = sparsediag_to_array(W[2], Ws[2])
B1_arr = sparsediag_to_array(B[1], Bs[1])
B2_arr = sparsediag_to_array(B[2], Bs[2])
