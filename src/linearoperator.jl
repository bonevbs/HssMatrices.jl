### Convenience access to Linear Operators
# Mostly copied over from LowRankApprox.jl

abstract type AbstractLinearOperator{T} end
const AbstractLinOp = AbstractLinearOperator
const AbstractMatOrLinOp{T} = Union{AbstractMatrix{T}, AbstractLinOp{T}}

mutable struct LinearOperator{T} <: AbstractLinOp{T}
  m::Int
  n::Int
  mul!::Function
  mulc!::Function
  getidx::Function
  _tmp::Union{Array{T}, Nothing}
end
const LinOp = LinearOperator

mutable struct HermitianLinearOperator{T} <: AbstractLinOp{T}
  n::Int
  mul!::Function
  getidx::Function
  _tmp::Union{Array{T}, Nothing}
end
const HermLinOp = HermitianLinearOperator

function LinOp(A)
  ishermitian(A) && return HermLinOp(A)
  T = eltype(A)
  m, n = size(A)
  ml!  = (y, _, x) ->  mul!(y, A, x)
  mulc! = (y, _, x) -> mul!(y, A', x)
  getidx = (i, j) -> A[i,j]
  LinOp{T}(m, n, ml!, mulc!, getidx, nothing)
end

function HermLinOp(A)
  T = eltype(A)
  m, n = size(A)
  m == n || throw(DimensionMismatch)
  ml! = (y, _, x) ->  mul!(y, A, x)
  getidx = (i, j) -> A[i,j]
  HermLinOp{T}(n, ml!, getidx, nothing)
end

convert(::Type{Array}, A::AbstractLinOp) = Matrix(A)
convert(::Type{Array{T}}, A::AbstractLinOp) where {T} = convert(Array{T}, Matrix(A))
convert(::Type{LinOp}, A::HermLinOp) = convert(LinOp{eltype(A)}, A)
convert(::Type{LinOp{T}}, A::HermLinOp{T}) where T = LinOp{T}(A.n, A.n, A.mul!, A.mul!, A.getidx, A._tmp)

adjoint(A::LinOp{T}) where {T} = LinOp{T}(A.n, A.m, A.mulc!, A.mul!, (i,j) -> conj(A.getidx(j,i)), nothing)
adjoint(A::HermLinOp) = A

eltype(::AbstractLinOp{T}) where {T} = T

Matrix(A::AbstractLinOp{T}) where {T} = A*Matrix{T}(I, size(A)...)

getindex(A::AbstractLinOp, ::Colon, ::Colon) = Matrix(A)
getindex(A::AbstractLinOp, rows, ::Colon) = A.getidx(rows, 1:A.n)
getindex(A::AbstractLinOp, ::Colon, cols) = A.getidx(1:A.m, cols)
getindex(A::AbstractLinOp, rows, cols) = A.getidx(rows, cols)

ishermitian(::LinOp) = false
ishermitian(::HermLinOp) = true
issymmetric(::LinOp) = false
issymmetric(A::HermLinOp) = isreal(A)

isreal(::AbstractLinOp{T}) where {T} = T <: Real

size(A::LinOp) = (A.m, A.n)
size(A::LinOp, dim::Integer) = dim == 1 ? A.m : (dim == 2 ? A.n : 1)
size(A::HermLinOp) = (A.n, A.n)
size(A::HermLinOp, dim::Integer) = (dim == 1 || dim == 2) ? A.n : 1

function transpose(A::LinOp{T}) where T
  n, m = size(A)
  mul!  = (y, L, x) -> (A.mulc!(y, L, conj(x)); conj!(y))
  mulc! = (y, L, x) -> ( A.mul!(y, L, conj(x)); conj!(y))
  getidx = (i,j) -> A.getidx(j,i)
  LinOp{T}(m, n, mul!, mulc!, getidx, nothing)
end
function transpose(A::HermLinOp{T}) where T
  n = size(A, 1)
  mul! = (y, L, x) -> (A.mul!(y, L, conj(x)); conj!(y))
  getidx = (i,j) -> A.getidx(j,i)
  HermLinOp{T}(n, mul!, getidx, nothing)
end

# matrix multiplication
^(A::AbstractLinOp, p::Integer) = Base.power_by_squaring(A, p)

mul!(C, A::AbstractLinOp, B::AbstractVecOrMat) = A.mul!(C, A, B)
mul!(C, A::Adjoint{<:Any,<:LinOp}, B::AbstractVecOrMat) = parent(A).mulc!(C, parent(A), B)
mul!(C, A::Adjoint{<:Any,<:HermLinOp}, B::AbstractVecOrMat) = mul!(C, parent(A), B)
mul!(C, A::AbstractMatrix, B::Adjoint{<:Any,<:AbstractLinOp}) = adjoint!(C, parent(B)*A')
mul!(C, A::AbstractMatrix, B::AbstractLinOp) = adjoint!(C, B'*A')


*(A::AbstractLinOp{T}, x::AbstractVector) where {T} =
  (y = Array{T}(undef, size(A,1)); mul!(y, A, x))
*(A::AbstractLinOp{T}, B::AbstractMatrix) where {T} =
  (C = Array{T}(undef, size(A,1), size(B,2)); mul!(C, A, B))
*(A::AbstractMatrix, B::AbstractLinOp{T}) where {T} =
  (C = Array{T}(undef, size(A,1), size(B,2)); mul!(C, A, B))

# scalar multiplication/division
for (f, g) in ((:(A::LinOp), :(c::Number)), (:(c::Number), :(A::LinOp)))
  @eval begin
    function *($f, $g)
      T = eltype(A)
      m, n = size(A)
      sc_mul!  = (y, _, x) -> ( mul!(y, A, x); lmul!(c, y))
      sc_mulc! = (y, _, x) -> (mul!(y, A', x); lmul!(c, y))
      sc_getidx = (i,j) -> c*A.getidx(i,j)
      LinOp{T}(m, n, sc_mul!, sc_mulc!, sc_getidx, nothing)
    end
  end
end

for (f, g) in ((:(A::HermLinOp), :(c::Number)), (:(c::Number), :(A::HermLinOp)))
  @eval begin
    function *($f, $g)
      T = eltype(A)
      n = size(A, 1)
      sc_mul! = (y, _, x) -> (mul!(y, A, x); lmul!(c, y))
      sc_getidx = (i,j) -> c*A.getidx(i,j)
      HermLinOp{T}(n, sc_mul!, sc_getidx, nothing)
    end
  end
end

-(A::AbstractLinOp) = -1*A

/(A::AbstractLinOp, c::Number) = A*(1/c)
\(c::Number, A::AbstractLinOp) = (1/c)*A

# operator addition/subtraction

for (f, a) in ((:+, 1), (:-, -1))
  @eval begin
    function $f(A::AbstractLinOp{T}, B::AbstractLinOp{T}) where T
      size(A) == size(B) || throw(DimensionMismatch)
      m, n = size(A)
      alpha = T($a)
      mul!  =  gen_linop_axpy(A, B, alpha)
      mulc! = gen_linop_axpyc(A, B, alpha)
      getidx = A.getidx(i,j) + B.getidx(i,j)
      LinOp{T}(m, n, mul!, mulc!, getidx, nothing)
    end

    function $f(A::HermLinOp{T}, B::HermLinOp{T}) where T
      size(A) == size(B) || throw(DimensionMismatch)
      n = size(A, 1)
      alpha = T($a)
      mul! = gen_linop_axpy(A, B, alpha)
      getidx = A.getidx(i,j) + B.getidx(i,j)
      HermLinOp{T}(n, mul!, getidx, nothing)
    end
  end
end

function gen_linop_axpy(A::AbstractLinOp{T}, B::AbstractLinOp{T}, alpha::T) where T
  function linop_axpy!(
      y::AbstractVecOrMat{T}, L::AbstractLinOp{T}, x::AbstractVecOrMat{T}) where T
    if isnull(L._tmp) || size(get(L._tmp)) != size(y)
      L._tmp = similar(y)
    end
    tmp = get(L._tmp)
    mul!( y , A, x)
    mul!(tmp, B, x)
    BLAS.axpy!(alpha*one(T), tmp, y)
  end
end


function gen_linop_axpyc(A::AbstractLinOp{T}, B::AbstractLinOp{T}, alpha::T) where T
  function linop_axpyc!(
    y::AbstractVecOrMat{T}, L::AbstractLinOp{T}, x::AbstractVecOrMat{T}) where T
    if isnull(L._tmp) || size(get(L._tmp)) != size(y)
      L._tmp = similar(y)
    end
    tmp = get(L._tmp)
    mul!( y , A', x)
    mul!(tmp, B', x)
    BLAS.axpy!(alpha*one(T), tmp, y)
  end
end

# # operator composition
# function *(A::AbstractLinOp{T}, B::AbstractLinOp{T}) where T
#   mA, nA = size(A)
#   mB, nB = size(B)
#   nA == mB || throw(DimensionMismatch)
#   mul!  =  gen_linop_comp(A, B)
#   mulc! = gen_linop_compc(A, B)
#   LinOp{T}(mA, nB, mul!, mulc!, nothing)
# end


# function gen_linop_comp(A::AbstractLinOp{T}, B::AbstractLinOp{T}) where T
#   function linop_comp!(
#       y::AbstractVector{T}, L::AbstractLinOp{T}, x::AbstractVector{T}) where T
#     n = size(B, 1)
#     if isnull(L._tmp) || length(get(L._tmp)) != n
#       L._tmp = Array{T}(undef, n)
#     end
#     tmp = get(L._tmp)
#     mul!(tmp, B,  x )
#     mul!( y , A, tmp)
#   end
#   function linop_comp!(
#       Y::AbstractMatrix{T}, L::AbstractLinOp{T}, X::AbstractMatrix{T}) where T
#     m = size(B, 1)
#     n = size(X, 2)
#     if isnull(L._tmp) || size(get(L._tmp)) != (m, n)
#       L._tmp = Array{T}(undef, m, n)
#     end
#     tmp = get(L._tmp)
#     mul!(tmp, B,  X )
#     mul!( Y , A, tmp)
#   end
# end


# function gen_linop_compc(A::AbstractLinOp{T}, B::AbstractLinOp{T}) where T
#   function linop_compc!(
#     y::AbstractVector{T}, L::AbstractLinOp{T}, x::AbstractVector{T}) where T
#     n = size(B, 1)
#     if isnull(L._tmp) || length(get(L._tmp)) != n
#       L._tmp = Array{T}(undef, n)
#     end
#     tmp = get(L._tmp)
#     mul!(tmp, B',  x )
#     mul!( y , A', tmp)
#   end
#   function linop_compc!(
#     Y::AbstractMatrix{T}, L::AbstractLinOp{T}, X::AbstractMatrix{T}) where T
#     m = size(B, 1)
#     n = size(X, 2)
#     if isnull(L._tmp) || size(get(L._tmp)) != (m, n)
#       L._tmp = Array{T}(undef, m, n)
#     end
#     tmp = get(L._tmp)
#     mul!(tmp, B',  X )
#     mul!( Y , A', tmp)
#   end
# end