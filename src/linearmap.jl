### Convenience access to Linear Operators
# Mostly copied over from LowRankApprox.jl
# This extends the linear operators from LowRankApprox to include custom getindex routines
# We call these structs LinearMaps

const AbstractLinOp = AbstractLinearOperator
const AbstractMatOrLinOp{T} = Union{AbstractMatrix{T}, AbstractLinOp{T}}

  mutable struct LinearMap{T, Mul<:Function, Mulc<:Function, GetIdx<:Function} <: AbstractLinOp{T}
    m::Int
    n::Int
    mul!::Mul
    mulc!::Mulc
    getidx::GetIdx
    LinearMap{T}(m, n, mul!, mulc!, getidx) where T = new{T, typeof(mul!), typeof(mulc!), typeof(getidx)}(m, n, mul!, mulc!, getidx)
  end
const LinMap = LinearMap

mutable struct HermitianLinearMap{T, Mul<:Function, GetIdx<:Function} <: AbstractLinOp{T}
  n::Int
  mul!::Mul
  getidx::GetIdx
  HermitianLinearMap{T}(m, n, mul!, getidx) where T = new{T, typeof(mul!), typeof(getidx)}(m, n, mul!, getidx)
end
const HermLinMap = HermitianLinearMap

function LinMap(A)
  ishermitian(A) && return HermLinMap(A)
  T = eltype(A)
  m, n = size(A)
  ml!  = (y, _, x) ->  mul!(y, A, x)
  mulc! = (y, _, x) -> mul!(y, A', x)
  getidx = (i, j) -> A[i,j]
  LinMap{T}(m, n, ml!, mulc!, getidx)
end

function HermLinMap(A)
  T = eltype(A)
  m, n = size(A)
  m == n || throw(DimensionMismatch)
  ml! = (y, _, x) ->  mul!(y, A, x)
  getidx = (i, j) -> A[i,j]
  HermLinMap{T}(n, ml!, getidx)
end

convert(::Type{LinMap}, A::HermLinMap) = convert(LinMap{eltype(A)}, A)
convert(::Type{LinMap{T}}, A::HermLinMap{T}) where T = LinMap{T}(A.n, A.n, A.mul!, A.mul!, A.getidx)

adjoint(A::LinMap{T}) where {T} = LinMap{T}(A.n, A.m, A.mulc!, A.mul!, (i,j) -> conj(A.getidx(j,i)))
adjoint(A::HermLinMap) = A


#Matrix(A::AbstractLinOp{T}) where {T} = A*Matrix{T}(I, size(A)...)

getindex(A::LinMap, ::Colon, ::Colon) = A.getidx(1:A.m, 1:A.n)
getindex(A::LinMap, rows, ::Colon) = A.getidx(rows, 1:A.n)
getindex(A::LinMap, ::Colon, cols) = A.getidx(1:A.m, cols)
getindex(A::LinMap, rows, cols) = A.getidx(rows, cols)

getindex(A::HermLinMap, ::Colon, ::Colon) = A.getidx(1:A.m, 1:A.n)
getindex(A::HermLinMap, rows, ::Colon) = A.getidx(rows, 1:A.n)
getindex(A::HermLinMap, ::Colon, cols) = A.getidx(1:A.m, cols)
getindex(A::HermLinMap, rows, cols) = A.getidx(rows, cols)

ishermitian(::LinMap) = false
ishermitian(::HermLinMap) = true
issymmetric(::LinMap) = false
issymmetric(A::HermLinMap) = isreal(A)

size(A::LinMap) = (A.m, A.n)
size(A::LinMap, dim::Integer) = dim == 1 ? A.m : (dim == 2 ? A.n : 1)
size(A::HermLinMap) = (A.n, A.n)
size(A::HermLinMap, dim::Integer) = (dim == 1 || dim == 2) ? A.n : 1

function transpose(A::LinMap{T}) where T
  n, m = size(A)
  mul!  = (y, L, x) -> (A.mulc!(y, L, conj(x)); conj!(y))
  mulc! = (y, L, x) -> ( A.mul!(y, L, conj(x)); conj!(y))
  getidx = (i,j) -> A.getidx(j,i)
  LinMap{T}(m, n, mul!, mulc!, getidx)
end
function transpose(A::HermLinMap{T}) where T
  n = size(A, 1)
  mul! = (y, L, x) -> (A.mul!(y, L, conj(x)); conj!(y))
  getidx = (i,j) -> A.getidx(j,i)
  HermLinMap{T}(n, mul!, getidx)
end

# # copied over from LowRankApprox.jl
# mul!(C, A::LinMap, B::AbstractVecOrMat) = A.mul!(C, A, B)
# mul!(C, A::Adjoint{<:Any,<:LinMap}, B::AbstractVecOrMat) = parent(A).mulc!(C, parent(A), B)
# mul!(C, A::Adjoint{<:Any,<:HermLinMap}, B::AbstractVecOrMat) = mul!(C, parent(A), B)
# mul!(C, A::AbstractMatrix, B::Adjoint{<:Any,<:AbstractLinMap}) = adjoint!(C, parent(B)*A')
# mul!(C, A::AbstractMatrix, B::AbstractLinMap) = adjoint!(C, B'*A')

# *(A::LinMap{T}, x::AbstractVector) where T = (y = Array{T}(undef, size(A,1)); mul!(y, A, x))
# *(A::LinMap{T}, B::AbstractMatrix) where T = (C = Array{T}(undef, size(A,1), size(B,2)); mul!(C, A, B))
# *(A::AbstractMatrix, B::LinMap{T}) where T = (C = Array{T}(undef, size(A,1), size(B,2)); mul!(C, A, B))

# scalar multiplication/division
for (f, g) in ((:(A::LinMap), :(c::Number)), (:(c::Number), :(A::LinMap)))
  @eval begin
    function *($f, $g)
      T = eltype(A)
      m, n = size(A)
      sc_mul!  = (y, _, x) -> ( mul!(y, A, x); lmul!(c, y))
      sc_mulc! = (y, _, x) -> (mul!(y, A', x); lmul!(c, y))
      sc_getidx = (i,j) -> c*A.getidx(i,j)
      LinMap{T}(m, n, sc_mul!, sc_mulc!, sc_getidx)
    end
  end
end

for (f, g) in ((:(A::HermLinMap), :(c::Number)), (:(c::Number), :(A::HermLinMap)))
  @eval begin
    function *($f, $g)
      T = eltype(A)
      n = size(A, 1)
      sc_mul! = (y, _, x) -> (mul!(y, A, x); lmul!(c, y))
      sc_getidx = (i,j) -> c*A.getidx(i,j)
      HermLinMap{T}(n, sc_mul!, sc_getidx)
    end
  end
end


# operator addition/subtraction
for (f, a) in ((:+, 1), (:-, -1))
  @eval begin
    function $f(A::HermLinMap{T}, B::HermLinMap{T}) where T
      size(A) == size(B) || throw(DimensionMismatch)
      n = size(A, 1)
      alpha = T($a)
      mul! = gen_linop_axpy(A, B, alpha)
      getidx = A.getidx(i,j) + B.getidx(i,j)
      HermLinMap{T}(n, mul!, getidx)
    end
  end
end